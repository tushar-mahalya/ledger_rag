"""FinQA benchmark through the production retrieval and generation code paths.

FinQA (Chen et al., EMNLP 2021, MIT license) contains numerical-reasoning questions
written by finance experts over real S&P 500 earnings-report pages (prose + tables).

Corpus: every unique report page from train, dev and test is ingested, so held-out
test questions compete against thousands of real distractor pages.
Questions: the test split only. Nothing is tuned on test.

Retrieval scopes (see docs/BENCHMARKS.md):
- ``global``: open retrieval over the whole corpus. Many FinQA questions do not name
  the company, so this is an intentionally harsh lower bound.
- ``company``: retrieval scoped to the filer, as a product would after the caller or
  chat session selects a company. Scoping reuses the production ACL group filter.

Answer accuracy runs the real RagService harness (system prompt, citation validation,
calculator tool loop) against any OpenAI- or Anthropic-compatible endpoint.
"""

import argparse
import asyncio
import json
import math
import random
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from ledger_rag.core.config import Settings
from ledger_rag.core.contracts import Retriever
from ledger_rag.core.models import Answer, Chunk, Document, Hit, Principal, Profile, QueryRequest
from ledger_rag.ingestion.chunking import chunk_document

TENANT = "finqa"
SOURCE = "https://github.com/czyssrs/FinQA"
DATASET_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
ANSWER_FORMAT = " Begin the answer with the final numeric result."


@dataclass
class Case:
    id: str
    question: str
    document_id: str
    company: str
    gold_units: list[str]
    answer: float | str
    program: str


@dataclass
class Corpus:
    documents: dict[str, Document] = field(default_factory=dict)
    units: dict[str, dict[str, str]] = field(default_factory=dict)
    companies: set[str] = field(default_factory=set)


def _clean(cell: str) -> str:
    return re.sub(r"\s+", " ", cell.replace("|", "/")).strip()


def table_markdown(table: list[list[str]]) -> tuple[str, list[str]]:
    """Render a FinQA table as markdown; returns the table and each rendered row."""
    if not table:
        return "", []
    width = max(len(row) for row in table)
    rows = [[_clean(c) for c in row] + [""] * (width - len(row)) for row in table]
    rendered = ["| " + " | ".join(row) + " |" for row in rows]
    separator = "|" + "---|" * width
    return "\n".join([rendered[0], separator, *rendered[1:]]), rendered


def page_document(example: dict[str, Any]) -> tuple[Document, dict[str, str]]:
    filename = example["filename"]
    company, year, page = filename.split("/")[:3]
    page_number = re.sub(r"\D", "", page) or "0"
    texts = list(example["pre_text"]) + list(example["post_text"])
    table, rows = table_markdown(example["table"])
    pre = [t.strip() for t in example["pre_text"] if t.strip() not in {"", "."}]
    post = [t.strip() for t in example["post_text"] if t.strip() not in {"", "."}]
    body = "\n\n".join(part for part in [" ".join(pre), table, " ".join(post)] if part)
    units = {f"text_{i}": t.strip() for i, t in enumerate(texts) if t.strip()}
    units.update({f"table_{i}": row for i, row in enumerate(rows)})
    document = Document(
        id=filename,
        tenant_id=TENANT,
        title=f"{company} annual report {year}, page {page_number}",
        text=body or ".",
        source=f"{SOURCE}/blob/{DATASET_COMMIT}/dataset",
        # FinQA gives the report year only; this is an approximate publication date.
        published_at=date(int(year), 12, 31),
        allowed_groups=[f"company:{company}"],
    )
    return document, units


def load(data_dir: Path) -> tuple[Corpus, list[Case]]:
    corpus = Corpus()
    cases: list[Case] = []
    for split in ["train", "dev", "test"]:
        for example in json.loads((data_dir / f"{split}.json").read_text()):
            document, units = page_document(example)
            corpus.documents.setdefault(document.id, document)
            corpus.units.setdefault(document.id, units)
            corpus.companies.add(document.id.split("/")[0])
            if split != "test":
                continue
            qa = example["qa"]
            gold = [
                corpus.units[document.id][key]
                for key in qa.get("gold_inds", {})
                if key in corpus.units[document.id]
            ]
            answer = qa["exe_ans"]
            cases.append(
                Case(
                    id=example["id"],
                    question=qa["question"],
                    document_id=document.id,
                    company=document.id.split("/")[0],
                    gold_units=gold,
                    answer=answer if isinstance(answer, str) else float(answer),
                    program=qa.get("program", ""),
                )
            )
    return corpus, cases


def principal(case: Case, scope: str, corpus: Corpus) -> Principal:
    companies = [case.company] if scope == "company" else sorted(corpus.companies)
    return Principal(
        subject="finqa-bench", tenant_id=TENANT, groups=[f"company:{c}" for c in companies]
    )


def page_metrics(
    hits: list[Hit], case: Case, ks: tuple[int, ...] = (1, 3, 5, 10)
) -> dict[str, float]:
    pages: list[str] = []
    for hit in hits:
        if hit.chunk.document_id not in pages:
            pages.append(hit.chunk.document_id)
    rank = pages.index(case.document_id) + 1 if case.document_id in pages else None
    metrics = {f"page_hit@{k}": float(rank is not None and rank <= k) for k in ks}
    metrics["page_mrr@10"] = 1 / rank if rank and rank <= 10 else 0.0
    if case.gold_units:
        for k in (3, 5):
            texts = [h.chunk.text for h in hits[:k] if h.chunk.document_id == case.document_id]
            found = sum(1 for unit in case.gold_units if any(unit in text for text in texts))
            metrics[f"evidence_recall@{k}"] = found / len(case.gold_units)
    return metrics


_NUMBER = re.compile(r"(?<![\w.])[-−]?\$?\s?\d[\d,]*(?:\.\d+)?")


def numbers(text: str) -> list[float]:
    text = re.sub(r"\[C\d+\]", " ", text)
    values = []
    for match in _NUMBER.findall(text):
        cleaned = re.sub(r"[$,\s]", "", match).replace("−", "-")
        try:
            values.append(float(cleaned))
        except ValueError:
            continue
    return values


def answer_correct(text: str, gold: float | str, *, rel_tol: float = 0.01) -> bool:
    """FinQA-style execution match on the first number, tolerant to percent/ratio scale."""
    if isinstance(gold, str):
        return text.strip().lower().startswith(gold.lower())
    values = numbers(text)
    if not values:
        return False
    value = values[0]
    for target in (gold, gold * 100, gold / 100):
        if math.isclose(value, target, rel_tol=rel_tol, abs_tol=0.005):
            return True
        if round(value, 1) == round(target, 1) and abs(target) >= 0.05:
            return True
    return False


def summarize(rows: list[dict[str, float]], seed: int = 7, samples: int = 1000) -> dict[str, Any]:
    keys = sorted({key for row in rows for key in row})
    rng = random.Random(seed)
    out: dict[str, Any] = {}
    for key in keys:
        values = [row[key] for row in rows if key in row]
        mean = sum(values) / len(values)
        boot = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(samples))
        out[key] = {
            "mean": round(mean, 4),
            "ci95": [round(boot[int(0.025 * samples)], 4), round(boot[int(0.975 * samples)], 4)],
            "n": len(values),
        }
    return out


def latency(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}
    return {
        "p50_ms": round(statistics.median(ordered), 1),
        "p95_ms": round(ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)], 1),
        "mean_ms": round(sum(ordered) / len(ordered), 1),
    }


def bench_settings(args: argparse.Namespace, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "demo_token": SecretStr("benchmark-only-token"),
        "retrieval_backend": "qdrant",
        "qdrant_url": args.qdrant_url,
        "qdrant_collection": args.collection,
        "embedding_base_url": args.embedding_url,
        "embedding_api_key": SecretStr(args.embedding_key),
        "embedding_model": args.embedding_model,
        "embedding_dimensions": args.embedding_dimensions,
        "embedding_query_prefix": args.query_prefix,
        "llm_timeout_seconds": 120,
        "query_deadline_seconds": 120,
        "reranker_timeout_seconds": 60,
        "rerank_candidates": args.rerank_candidates,
    }
    values.update(overrides)
    return Settings(**values)


async def ingest(args: argparse.Namespace, corpus: Corpus) -> dict[str, Any]:
    from ledger_rag.retrieval.qdrant import QdrantRetriever

    store = QdrantRetriever(bench_settings(args))
    start = time.perf_counter()
    count = 0
    batch: list[Chunk] = []
    try:
        for document in corpus.documents.values():
            batch.extend(chunk_document(document))
            if len(batch) >= 64:
                await store.upsert(batch)
                count += len(batch)
                batch = []
                print(f"\ringested {count} chunks", end="", flush=True)
        if batch:
            await store.upsert(batch)
            count += len(batch)
    finally:
        await store.close()
    print()
    return {
        "documents": len(corpus.documents),
        "chunks": count,
        "seconds": round(time.perf_counter() - start, 1),
    }


def make_store(args: argparse.Namespace, fusion: str, rerank: bool) -> Retriever:
    from ledger_rag.retrieval.qdrant import QdrantRetriever
    from ledger_rag.retrieval.reranking import RerankingRetriever

    settings = bench_settings(
        args, retrieval_fusion=fusion, reranker_url=args.reranker_url if rerank else ""
    )
    store: Retriever = QdrantRetriever(settings)
    return RerankingRetriever(store, settings) if rerank else store


async def retrieval(args: argparse.Namespace, corpus: Corpus, cases: list[Case]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for config in args.configs:
        fusion, _, rerank_flag = config.partition(":")
        rerank = rerank_flag == "rerank"
        name = f"{fusion}{'+rerank' if rerank else ''}"
        store = make_store(args, fusion, rerank)
        try:
            for scope in args.scopes:
                rows: list[dict[str, float]] = []
                times: list[float] = []
                for index, case in enumerate(cases):
                    query = QueryRequest(question=case.question, top_k=10)
                    start = time.perf_counter()
                    hits = await store.search(query, principal(case, scope, corpus))
                    times.append((time.perf_counter() - start) * 1000)
                    rows.append(page_metrics(hits, case))
                    if index % 50 == 0:
                        print(f"\r{name}/{scope}: {index}/{len(cases)}", end="", flush=True)
                print()
                results[f"{name}/{scope}"] = {"metrics": summarize(rows), "latency": latency(times)}
                brief = {k: v["mean"] for k, v in results[f"{name}/{scope}"]["metrics"].items()}
                print(name, scope, json.dumps(brief), json.dumps(latency(times)))
        finally:
            await store.close()
    return results


async def answers(args: argparse.Namespace, corpus: Corpus, cases: list[Case]) -> dict[str, Any]:
    from ledger_rag.generation.providers import make_provider
    from ledger_rag.orchestration.service import RagService

    settings = bench_settings(
        args,
        provider=args.provider,
        llm_base_url=args.llm_url,
        llm_api_key=SecretStr(args.llm_key),
        llm_model=args.llm_model,
        reranker_url=args.reranker_url if args.rerank_answers else "",
        max_tool_rounds=3,
    )
    provider = make_provider(settings)
    store = make_store(args, "hybrid", args.rerank_answers)
    service = RagService(settings, store, provider)
    sample = random.Random(args.seed).sample(cases, min(args.answer_samples, len(cases)))
    records: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(sample):
            for mode in args.answer_modes:
                query = QueryRequest(
                    question=case.question + ANSWER_FORMAT,
                    profile=Profile(args.profile),
                    top_k=args.answer_top_k,
                )
                who = principal(case, "company", corpus)
                start = time.perf_counter()
                if mode == "oracle":
                    hits = [
                        Hit(chunk=chunk, score=1.0)
                        for chunk in chunk_document(corpus.documents[case.document_id])
                    ]
                    retrieval_ms = 0.0
                else:
                    hits, retrieval_ms = await service.retrieve(query, who)
                error = ""
                answer: Answer | None = None
                try:
                    answer = await service.generate(
                        query, hits, retrieval_ms, f"finqa-{index}", who
                    )
                except Exception as exc:  # record upstream failures instead of aborting
                    error = type(exc).__name__
                total_ms = (time.perf_counter() - start) * 1000
                text = answer.answer if answer and not answer.abstained else ""
                records.append(
                    {
                        "id": case.id,
                        "mode": mode,
                        "question": case.question,
                        "gold": case.answer,
                        "answer": text[:600],
                        "correct": bool(text) and answer_correct(text, case.answer),
                        "abstained": answer.abstained if answer else True,
                        "error": error,
                        "gold_page_in_context": any(
                            h.chunk.document_id == case.document_id for h in hits
                        ),
                        "tool_calls": answer.tool_calls if answer else 0,
                        "retrieval_ms": round(retrieval_ms, 1),
                        "generation_ms": answer.generation_ms if answer else 0.0,
                        "total_ms": round(total_ms, 1),
                    }
                )
            first = [r for r in records if r["mode"] == args.answer_modes[0]]
            accuracy = sum(r["correct"] for r in first) / len(first)
            print(f"\ranswers {index + 1}/{len(sample)} acc={accuracy:.3f}", end="", flush=True)
        print()
    finally:
        await store.close()
        await provider.close()
    summary: dict[str, Any] = {}
    for mode in args.answer_modes:
        subset = [r for r in records if r["mode"] == mode]
        rows = [
            {
                "execution_accuracy": float(r["correct"]),
                "abstention_rate": float(r["abstained"]),
                "error_rate": float(bool(r["error"])),
                "gold_page_in_context": float(r["gold_page_in_context"]),
                "tool_call_rate": float(r["tool_calls"] > 0),
            }
            for r in subset
        ]
        answered = [r for r in subset if not r["abstained"]]
        summary[mode] = {
            "metrics": summarize(rows),
            "precision_when_answered": (
                round(sum(r["correct"] for r in answered) / len(answered), 4) if answered else None
            ),
            "latency_total": latency([r["total_ms"] for r in subset]),
            "latency_generation": latency([r["generation_ms"] for r in subset]),
        }
    return {"summary": summary, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="FinQA benchmark on production code paths")
    parser.add_argument("command", choices=["stats", "ingest", "retrieval", "answers"])
    parser.add_argument("--data", type=Path, default=Path("data/finqa"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="finqa_bench")
    parser.add_argument("--embedding-url", default="http://localhost:8080/v1")
    parser.add_argument("--embedding-key", default="local-unused")
    parser.add_argument("--embedding-model", default="Alibaba-NLP/gte-modernbert-base")
    parser.add_argument("--embedding-dimensions", type=int, default=768)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--reranker-url", default="http://localhost:8081")
    parser.add_argument("--rerank-candidates", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="First N test cases (0 = all)")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["sparse", "dense", "hybrid", "hybrid:rerank"],
        help="fusion[:rerank] entries",
    )
    parser.add_argument("--scopes", nargs="+", default=["company", "global"])
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--llm-url", default="http://localhost:11434/v1")
    parser.add_argument("--llm-key", default="local-unused")
    parser.add_argument("--llm-model", default="qwen2.5:3b")
    parser.add_argument("--profile", choices=["chat", "voice"], default="chat")
    parser.add_argument("--answer-samples", type=int, default=100)
    parser.add_argument("--answer-top-k", type=int, default=3)
    parser.add_argument("--answer-modes", nargs="+", default=["retrieved", "oracle"])
    parser.add_argument("--rerank-answers", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    corpus, cases = load(args.data)
    if args.limit:
        cases = cases[: args.limit]
    report: dict[str, Any] = {
        "dataset": {
            "name": "FinQA",
            "license": "MIT",
            "commit": DATASET_COMMIT,
            "corpus_pages": len(corpus.documents),
            "corpus_chunks": sum(len(chunk_document(d)) for d in corpus.documents.values()),
            "companies": len(corpus.companies),
            "test_questions": len(cases),
        },
        "embedding_model": args.embedding_model,
        "reranker": args.reranker_url and "gte-reranker-modernbert-base",
    }
    if args.command == "ingest":
        report["ingest"] = asyncio.run(ingest(args, corpus))
    elif args.command == "retrieval":
        report["rerank_candidates"] = args.rerank_candidates
        report["retrieval"] = asyncio.run(retrieval(args, corpus, cases))
    elif args.command == "answers":
        report["llm"] = {
            "provider": args.provider,
            "model": args.llm_model,
            "profile": args.profile,
        }
        report["answers"] = asyncio.run(answers(args, corpus, cases))
    print(json.dumps({k: v for k, v in report.items() if k != "answers"}, indent=2)[:3000])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
