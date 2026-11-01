"""FinQA benchmark through the production retrieval and generation code paths.

FinQA (Chen et al., EMNLP 2021, MIT license) contains numerical-reasoning questions
written by finance experts over real S&P 500 earnings-report pages (prose + tables).

Corpus: every unique report page from train, dev and test is ingested, so held-out
test questions compete against thousands of real distractor pages.
Questions: select train/dev/test with --split (default dev). The test split is
sealed: selecting it requires --unseal-test and is recorded in the report. Use dev
for architecture selection; held-out test is reporting-only.

Retrieval scopes (see docs/EVALUATION.md):
- ``global``: open retrieval over the whole corpus. Many FinQA questions do not name
  the company, so this is an intentionally harsh lower bound.
- ``company``: retrieval scoped to the filer, as a product would after the caller or
  chat session selects a company. Scoping reuses the production ACL group filter.

Answer accuracy (see finqa_answers.py) sends retrieved or oracle context to an
OpenAI-compatible endpoint with a structured final-answer contract and scores only
the parsed ``ANSWER:`` line.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from ledger_rag.core.config import Settings
from ledger_rag.core.contracts import Retriever
from ledger_rag.core.models import Chunk, Document, Hit, Principal, QueryRequest
from ledger_rag.evaluation.evidence import DocumentEvidence, evidence_metrics, map_units
from ledger_rag.evaluation.uncertainty import latency, paired_cluster_difference, summarize_rows
from ledger_rag.ingestion.chunking import chunk_document

TENANT = "finqa"
SOURCE = "https://github.com/czyssrs/FinQA"
DATASET_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
SPLITS = ("train", "dev", "test")
# Deterministic company folds: ~20% of companies form a company-disjoint holdout view.
HOLDOUT_FOLDS = 5


@dataclass
class Case:
    id: str
    question: str
    document_id: str
    company: str
    gold_units: list[str]
    answer: float | str
    program: str
    gold_keys: list[str] = field(default_factory=list)
    unknown_gold_keys: list[str] = field(default_factory=list)
    gold_inds_missing: bool = False
    answer_text: str = ""

    @property
    def report(self) -> str:
        """Bootstrap cluster: one annual report (``company/year``)."""
        return "/".join(self.document_id.split("/")[:2])


@dataclass
class Corpus:
    documents: dict[str, Document] = field(default_factory=dict)
    units: dict[str, dict[str, str]] = field(default_factory=dict)
    companies: set[str] = field(default_factory=set)
    split_reports: dict[str, set[str]] = field(default_factory=dict)
    split_companies: dict[str, set[str]] = field(default_factory=dict)
    _evidence: dict[str, DocumentEvidence] = field(default_factory=dict, repr=False)

    def evidence(self, document_id: str) -> DocumentEvidence:
        """Chunks rebuilt exactly as ingested, plus the gold-unit -> chunk mapping."""
        if document_id not in self._evidence:
            chunks = chunk_document(self.documents[document_id])
            self._evidence[document_id] = DocumentEvidence(
                chunks, map_units(self.units[document_id], chunks)
            )
        return self._evidence[document_id]


def corpus_sha256(corpus: Corpus) -> str:
    """Same fingerprint as bulk ingestion's ``corpus_sha256`` (documents in load order)."""
    digest = hashlib.sha256()
    for document in corpus.documents.values():
        digest.update(document.model_dump_json().encode())
        digest.update(b"\n")
    return digest.hexdigest()


def company_fold(company: str) -> int:
    return int(hashlib.sha256(company.encode()).hexdigest(), 16) % HOLDOUT_FOLDS


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


def load(data_dir: Path, question_split: str = "dev") -> tuple[Corpus, list[Case]]:
    """Index pages from every split; return questions from ``question_split`` only.

    Gold evidence keys absent from the page (``unknown_gold_keys``) and missing
    ``gold_inds`` are recorded on each case, never silently dropped; see
    :func:`gold_label_audit` and :func:`require_valid_gold`.
    """
    if question_split not in SPLITS:
        raise ValueError("question_split must be train, dev or test")
    corpus = Corpus()
    cases: list[Case] = []
    for split in SPLITS:
        reports = corpus.split_reports.setdefault(split, set())
        companies = corpus.split_companies.setdefault(split, set())
        for example in json.loads((data_dir / f"{split}.json").read_text()):
            document, units = page_document(example)
            corpus.documents.setdefault(document.id, document)
            corpus.units.setdefault(document.id, units)
            company = document.id.split("/")[0]
            corpus.companies.add(company)
            companies.add(company)
            reports.add("/".join(document.id.split("/")[:2]))
            if split != question_split:
                continue
            qa = example["qa"]
            page_units = corpus.units[document.id]
            gold_inds = qa.get("gold_inds")
            keys = list(gold_inds or {})
            answer = qa["exe_ans"]
            cases.append(
                Case(
                    id=example["id"],
                    question=qa["question"],
                    document_id=document.id,
                    company=company,
                    gold_units=[page_units[key] for key in keys if key in page_units],
                    answer=answer if isinstance(answer, str) else float(answer),
                    program=qa.get("program", ""),
                    gold_keys=[key for key in keys if key in page_units],
                    unknown_gold_keys=[key for key in keys if key not in page_units],
                    gold_inds_missing=gold_inds is None,
                    answer_text=str(qa.get("answer", "")),
                )
            )
    return corpus, cases


def gold_label_audit(cases: list[Case]) -> dict[str, Any]:
    unknown = [case for case in cases if case.unknown_gold_keys]
    return {
        "cases": len(cases),
        "gold_keys": sum(len(case.gold_keys) for case in cases),
        "unknown_gold_keys": sum(len(case.unknown_gold_keys) for case in cases),
        "cases_with_unknown_gold_keys": [case.id for case in unknown][:50],
        "cases_missing_gold_inds": sum(case.gold_inds_missing for case in cases),
        "cases_without_gold_keys": sum(not case.gold_keys for case in cases),
    }


def require_valid_gold(cases: list[Case]) -> None:
    """Fail loudly: evaluating with unresolvable gold labels would bias evidence metrics."""
    audit = gold_label_audit(cases)
    if audit["unknown_gold_keys"] or audit["cases_missing_gold_inds"]:
        raise ValueError(
            f"Invalid FinQA gold labels: {audit['unknown_gold_keys']} unknown keys, "
            f"{audit['cases_missing_gold_inds']} cases missing gold_inds "
            f"(e.g. {audit['cases_with_unknown_gold_keys'][:5]})"
        )


def evidence_audit(corpus: Corpus, cases: list[Case]) -> dict[str, Any]:
    methods: dict[str, int] = {}
    for case in cases:
        mapping = corpus.evidence(case.document_id).units
        for key in case.gold_keys:
            method = mapping[key].method
            methods[method] = methods.get(method, 0) + 1
    return {
        "method": "normalized unit-text containment in deterministically rebuilt chunks",
        "gold_units_by_mapping": methods,
    }


def holdout_views(corpus: Corpus, cases: list[Case]) -> dict[str, set[str]]:
    """Case-ID subsets for company/report-disjoint reporting.

    FinQA's native splits share companies and reports, so the default split is a
    question-level holdout only. ``*_unseen_in_train`` subsets are strictly disjoint
    from training pages' companies/reports; ``company_fold_holdout`` is a
    deterministic ~20% company partition for company-disjoint selection studies.
    """
    train_companies = corpus.split_companies.get("train", set())
    train_reports = corpus.split_reports.get("train", set())
    return {
        "all": {case.id for case in cases},
        "company_unseen_in_train": {c.id for c in cases if c.company not in train_companies},
        "report_unseen_in_train": {c.id for c in cases if c.report not in train_reports},
        "company_fold_holdout": {c.id for c in cases if company_fold(c.company) == 0},
        "company_fold_selection": {c.id for c in cases if company_fold(c.company) != 0},
    }


def principal(case: Case, scope: str, corpus: Corpus) -> Principal:
    companies = [case.company] if scope == "company" else sorted(corpus.companies)
    return Principal(
        subject="finqa-bench", tenant_id=TENANT, groups=[f"company:{c}" for c in companies]
    )


def page_metrics(
    hits: list[Hit],
    case: Case,
    evidence: DocumentEvidence | None = None,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float | None]:
    """Page hit/MRR plus gold-unit evidence metrics (see ledger_rag.evaluation.evidence)."""
    pages: list[str] = []
    for hit in hits:
        if hit.chunk.document_id not in pages:
            pages.append(hit.chunk.document_id)
    rank = pages.index(case.document_id) + 1 if case.document_id in pages else None
    metrics: dict[str, float | None] = {
        f"page_recall@{k}": float(rank is not None and rank <= k) for k in ks
    }
    metrics["page_mrr@10"] = 1 / rank if rank and rank <= 10 else 0.0
    if evidence is not None:
        gold = [evidence.units[key] for key in case.gold_keys]
        metrics.update(evidence_metrics([hit.chunk.id for hit in hits], gold, ks))
    return metrics


def summarize(
    rows: list[dict[str, float | None]],
    clusters: list[str] | None = None,
    seed: int = 7,
    samples: int = 2000,
) -> dict[str, Any]:
    """Report-cluster bootstrap; i.i.d. per-row resampling only if no clusters are given."""
    labels = clusters if clusters is not None else [str(i) for i in range(len(rows))]
    return summarize_rows(rows, labels, seed=seed, samples=samples)


def bench_settings(args: argparse.Namespace, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "demo_token": SecretStr("benchmark-only-token"),
        "retrieval_backend": "qdrant",
        "qdrant_url": args.qdrant_url,
        "qdrant_collection": args.collection,
        "embedding_base_url": args.embedding_url,
        "embedding_api_key": SecretStr(args.embedding_key),
        "embedding_model": args.embedding_model,
        "embedding_space_id": args.embedding_space_id,
        "embedding_dimensions": args.embedding_dimensions,
        "embedding_query_prefix": args.query_prefix,
        "qdrant_quantization": args.quantization,
        "qdrant_oversampling": args.oversampling,
        "llm_timeout_seconds": 120,
        "query_deadline_seconds": 120,
        "reranker_timeout_seconds": 60,
        "rerank_candidates": args.rerank_candidates,
    }
    qdrant_key = getattr(args, "qdrant_api_key", "") or os.environ.get("LEDGER_QDRANT_API_KEY", "")
    if qdrant_key:
        values["qdrant_api_key"] = SecretStr(qdrant_key)
    reranker_key = getattr(args, "reranker_key", "")
    if reranker_key:
        values["reranker_api_key"] = SecretStr(reranker_key)
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


def arm_name(config: str) -> tuple[str, bool, str]:
    fusion, _, rerank_flag = config.partition(":")
    if fusion not in {"sparse", "dense", "hybrid"} or rerank_flag not in {"", "rerank"}:
        raise ValueError(f"Unknown retrieval arm {config!r}")
    rerank = rerank_flag == "rerank"
    return fusion, rerank, f"{fusion}{'+rerank' if rerank else ''}"


async def search_all(
    args: argparse.Namespace,
    fusion: str,
    rerank: bool,
    queries: list[tuple[QueryRequest, Principal]],
    label: str,
) -> list[tuple[list[Hit], float]]:
    """Run queries with one store per worker (each store serializes its own requests)."""
    workers = max(1, min(int(getattr(args, "retrieval_concurrency", 1)), len(queries) or 1))
    results: list[tuple[list[Hit], float] | None] = [None] * len(queries)
    queue: asyncio.Queue[int] = asyncio.Queue()
    for index in range(len(queries)):
        queue.put_nowait(index)
    done = 0

    async def worker() -> None:
        nonlocal done
        store = make_store(args, fusion, rerank)
        try:
            while not queue.empty():
                index = queue.get_nowait()
                query, who = queries[index]
                start = time.perf_counter()
                hits = await store.search(query, who)
                results[index] = (hits, (time.perf_counter() - start) * 1000)
                done += 1
                if done % 100 == 0:
                    print(f"{label}: {done}/{len(queries)}", file=sys.stderr, flush=True)
        finally:
            await store.close()

    await asyncio.gather(*(worker() for _ in range(workers)))
    return [result for result in results if result is not None]


def _metric_values(rows: dict[str, dict[str, float | None]], metric: str) -> dict[str, float]:
    return {key: value for key, row in rows.items() if (value := row.get(metric)) is not None}


async def retrieval(args: argparse.Namespace, corpus: Corpus, cases: list[Case]) -> dict[str, Any]:
    """Every arm x scope on identical cases; report-cluster CIs and holdout views."""
    clusters = {case.id: case.report for case in cases}
    views = holdout_views(corpus, cases)
    seed = getattr(args, "seed", 13)
    samples = getattr(args, "bootstrap_samples", 2000)
    results: dict[str, Any] = {}
    per_case: dict[str, dict[str, dict[str, float | None]]] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    unknown_chunks = 0
    for config in args.configs:
        fusion, rerank, name = arm_name(config)
        for scope in args.scopes:
            label = f"{name}/{scope}"
            queries = [
                (QueryRequest(question=case.question, top_k=10), principal(case, scope, corpus))
                for case in cases
            ]
            started = time.perf_counter()
            outcomes = await search_all(args, fusion, rerank, queries, label)
            rows: list[dict[str, float | None]] = []
            times: list[float] = []
            arm_records: list[dict[str, Any]] = []
            for case, (hits, elapsed) in zip(cases, outcomes, strict=True):
                for hit in hits:
                    known = corpus.documents.get(hit.chunk.document_id) is not None and any(
                        chunk.id == hit.chunk.id
                        for chunk in corpus.evidence(hit.chunk.document_id).chunks
                    )
                    unknown_chunks += not known
                rows.append(page_metrics(hits, case, corpus.evidence(case.document_id)))
                times.append(elapsed)
                arm_records.append(
                    {
                        "id": case.id,
                        "hits": [f"{h.chunk.document_id}#{h.chunk.ordinal}" for h in hits],
                        "scores": [round(h.score, 5) for h in hits],
                    }
                )
            per_case[label] = {case.id: row for case, row in zip(cases, rows, strict=True)}
            records[label] = arm_records
            view_metrics = {}
            for view, members in views.items():
                if view == "all" or not members:
                    continue
                subset = [
                    (row, case.report)
                    for row, case in zip(rows, cases, strict=True)
                    if case.id in members
                ]
                view_metrics[view] = summarize(
                    [row for row, _ in subset], [c for _, c in subset], seed, samples
                )
            results[label] = {
                "metrics": summarize(rows, [case.report for case in cases], seed, samples),
                "views": view_metrics,
                "latency": latency(times),
                "wall_seconds": round(time.perf_counter() - started, 1),
            }
            brief = {
                key: value["mean"]
                for key, value in results[label]["metrics"].items()
                if key in {"page_recall@1", "page_recall@5", "page_mrr@10", "evidence_recall@5"}
            }
            print(label, json.dumps(brief), json.dumps(latency(times)), file=sys.stderr)
    comparisons: dict[str, Any] = {}
    for scope in args.scopes:
        base = f"hybrid/{scope}"
        if base not in per_case:
            continue
        for label in per_case:
            if label == base or not label.endswith(f"/{scope}"):
                continue
            comparisons[f"{label} - {base}"] = {
                metric: paired_cluster_difference(
                    _metric_values(per_case[base], metric),
                    _metric_values(per_case[label], metric),
                    clusters,
                    seed=seed,
                    samples=samples,
                )
                for metric in ("page_recall@5", "evidence_recall@5", "evidence_mrr@10")
            }
    return {
        "arms": results,
        "paired_differences": comparisons,
        "views": {view: len(members) for view, members in views.items()},
        "integrity": {"unknown_retrieved_chunk_ids": unknown_chunks},
        "records": records,
    }


def _git_sha() -> str:
    try:
        root = Path(__file__).resolve().parents[4]
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def evaluation_source_sha256() -> str:
    """Digest of the evaluation package source that produced a report."""
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in sorted(root.rglob("*.py")):
        name = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big") + name)
        digest.update(len(content).to_bytes(8, "big") + content)
    return digest.hexdigest()


def _temperature(value: str) -> float | None:
    if value == "default":
        return None
    temperature = float(value)
    if not 0 <= temperature <= 2:
        raise argparse.ArgumentTypeError("temperature must be within [0, 2] or 'default'")
    return temperature


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FinQA benchmark on production code paths")
    parser.add_argument(
        "command",
        choices=["stats", "ingest", "retrieval", "contexts", "answers"],
        help="contexts: save answer-retrieval hits so answers can run without GPU services",
    )
    parser.add_argument("--data", type=Path, default=Path("data/finqa"))
    parser.add_argument(
        "--split",
        choices=list(SPLITS),
        default="dev",
        help="Question split only; dev for comparisons. test is sealed (see --unseal-test)",
    )
    parser.add_argument(
        "--unseal-test",
        action="store_true",
        help="Required for --split test; only after configuration freeze. Logged in reports",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="finqa_bench")
    parser.add_argument("--embedding-url", default="http://localhost:8080/v1")
    parser.add_argument(
        "--embedding-key", default=os.environ.get("LEDGER_EMBEDDING_API_KEY", "local-unused")
    )
    parser.add_argument("--embedding-model", default="Alibaba-NLP/gte-modernbert-base")
    parser.add_argument("--embedding-revision", default="", help="Recorded for lineage only")
    parser.add_argument(
        "--embedding-space-id", default=os.environ.get("LEDGER_EMBEDDING_SPACE_ID", "")
    )
    parser.add_argument("--embedding-dimensions", type=int, default=768)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument(
        "--quantization",
        choices=["none", "scalar", "binary"],
        default="none",
        help="Must match the collection's effective dense-vector configuration",
    )
    parser.add_argument("--oversampling", type=float, default=2.0)
    parser.add_argument("--reranker-url", default="http://localhost:8081")
    parser.add_argument("--reranker-key", default=os.environ.get("LEDGER_RERANKER_API_KEY", ""))
    parser.add_argument("--reranker-model", default="Alibaba-NLP/gte-reranker-modernbert-base")
    parser.add_argument("--reranker-revision", default="", help="Recorded for lineage only")
    parser.add_argument("--rerank-candidates", type=int, default=20)
    parser.add_argument(
        "--limit", type=int, default=0, help="First N selected-split cases (0 = all)"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["sparse", "dense", "hybrid", "hybrid:rerank"],
        help="fusion[:rerank] entries",
    )
    parser.add_argument("--scopes", nargs="+", default=["company", "global"])
    parser.add_argument("--retrieval-concurrency", type=int, default=4)
    parser.add_argument(
        "--expected-corpus-sha256", default="", help="Fail unless the rebuilt corpus matches"
    )
    parser.add_argument(
        "--llm-url",
        default=os.environ.get("LEDGER_LLM_BASE_URL", "http://localhost:11434/v1"),
    )
    parser.add_argument("--llm-key", default=os.environ.get("LEDGER_LLM_API_KEY", "local-unused"))
    parser.add_argument(
        "--llm-models",
        nargs="+",
        default=["qwen2.5:3b"],
        help="model or model:reasoning_effort entries (ignored when arms are given)",
    )
    parser.add_argument(
        "--llm-arms",
        default="",
        help='JSON list of arms: [{"label", "model", "temperature", "max_tokens", '
        '"reasoning_effort", "top_p", "top_k", "presence_penalty", "seed", "extra_body"}]',
    )
    parser.add_argument("--llm-arms-file", type=Path, help="File holding --llm-arms JSON")
    parser.add_argument(
        "--llm-endpoint-kind",
        choices=["hosted", "self-hosted"],
        default="hosted",
        help="hosted: metered API, concurrency <= 8, optional --llm-rpm pacing; "
        "self-hosted: unpaced, concurrency <= 256, readiness probe and model check",
    )
    parser.add_argument(
        "--llm-model-revision",
        default="",
        help="Expected weights revision; self-hosted servers must serve '<model>@<revision>'",
    )
    parser.add_argument("--llm-concurrency", type=int, default=8)
    parser.add_argument(
        "--llm-rpm", type=float, default=None, help="Client-side requests/minute per model"
    )
    parser.add_argument(
        "--llm-stream",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stream responses (SSE); default: on for self-hosted, off for hosted",
    )
    parser.add_argument("--llm-request-timeout", type=float, default=120)
    parser.add_argument(
        "--llm-deadline", type=float, default=180, help="Per-completion budget incl. retries"
    )
    parser.add_argument(
        "--llm-ready-timeout",
        type=float,
        default=1200,
        help="Self-hosted only: wait this long for a scale-to-zero server to start",
    )
    parser.add_argument(
        "--contexts", type=Path, help="Saved 'contexts' report; answers then skip retrieval"
    )
    parser.add_argument(
        "--temperature",
        type=_temperature,
        default=0.0,
        help="Sampling temperature, or 'default' to omit it (Gemini 3 guidance)",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--llm-seed", type=int, default=None)
    parser.add_argument("--answer-samples", type=int, default=100)
    parser.add_argument(
        "--answer-sample-strategy",
        choices=["random", "stratified"],
        default="random",
        help="stratified: evidence source x program steps, proportional allocation",
    )
    parser.add_argument(
        "--answer-modes",
        nargs="+",
        default=["retrieved@3", "retrieved@5", "oracle"],
        help="oracle and/or retrieved@k entries",
    )
    parser.add_argument("--answer-retrieval", default="hybrid:rerank")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--git-sha", default="", help="Override detected Git revision")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.limit < 0:
        parser.error("--limit must be zero or positive")
    if args.split == "test" and not args.unseal_test:
        parser.error("the test split is sealed; pass --unseal-test after configuration freeze")
    if not 1 <= args.retrieval_concurrency <= 16:
        parser.error("--retrieval-concurrency must be between 1 and 16")
    from ledger_rag.evaluation.llm import MAX_CONCURRENCY

    limit = MAX_CONCURRENCY[args.llm_endpoint_kind]
    if not 1 <= args.llm_concurrency <= limit:
        parser.error(
            f"--llm-concurrency must be between 1 and {limit} for {args.llm_endpoint_kind} "
            "endpoints"
        )
    if args.llm_rpm is not None and args.llm_rpm <= 0:
        parser.error("--llm-rpm must be positive")
    if args.llm_endpoint_kind == "self-hosted" and args.llm_rpm is not None:
        parser.error("--llm-rpm paces hosted APIs only; self-hosted endpoints are unpaced")
    if args.llm_request_timeout <= 0 or args.llm_deadline <= 0 or args.llm_ready_timeout <= 0:
        parser.error("LLM timeouts must be positive")
    if args.llm_model_revision and not re.fullmatch(r"[0-9a-f]{40}", args.llm_model_revision):
        parser.error("--llm-model-revision must be a full 40-character commit sha")
    try:
        for config in args.configs:
            arm_name(config)
        from ledger_rag.evaluation.benchmarks.finqa_answers import generation_configs, parse_mode

        for mode in args.answer_modes:
            parse_mode(mode)
        arm_name(args.answer_retrieval)
        if args.command == "answers":
            generation_configs(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


def check_served_models(args: argparse.Namespace, served: list[str]) -> None:
    """Every arm's model must be served; a pinned revision must be advertised as an alias."""
    from ledger_rag.evaluation.benchmarks.finqa_answers import generation_configs

    for model in sorted({config.model for config in generation_configs(args)}):
        if model not in served:
            raise ValueError(f"LLM endpoint does not serve {model!r} (serves {served})")
        if args.llm_model_revision and f"{model}@{args.llm_model_revision}" not in served:
            raise ValueError(f"LLM endpoint does not advertise {model}@{args.llm_model_revision}")


def load_saved_contexts(
    path: Path, fingerprint: str, collection: str, split: str
) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    saved = json.loads(raw)
    if "lineage" not in saved:
        saved = {**saved["contexts"], "run_id": saved.get("run_id")}
    lineage = saved["lineage"]
    if lineage["corpus_sha256"] != fingerprint or lineage["collection"] != collection:
        raise ValueError("Saved contexts come from a different corpus or collection")
    if saved["dataset"]["question_split"] != split:
        raise ValueError("Saved contexts come from a different question split")
    return saved, hashlib.sha256(raw).hexdigest()


async def execute(args: argparse.Namespace) -> dict[str, Any]:
    """Build the report for one command; shared by the CLI and the Modal job."""
    if args.split == "test":
        print("WARNING: evaluating the sealed FinQA test split (--unseal-test)", file=sys.stderr)
    corpus, cases = load(args.data, args.split)
    split_questions = len(cases)
    if args.limit:
        cases = cases[: args.limit]
    fingerprint = corpus_sha256(corpus)
    if args.expected_corpus_sha256 and fingerprint != args.expected_corpus_sha256:
        raise ValueError("Rebuilt FinQA corpus does not match the expected ingestion fingerprint")
    report: dict[str, Any] = {
        "lineage": {
            "git_sha": args.git_sha or _git_sha(),
            "evaluation_source_sha256": evaluation_source_sha256(),
            "python": sys.version.split()[0],
            "collection": args.collection,
            "embedding_model": args.embedding_model,
            "embedding_revision": args.embedding_revision or None,
            "embedding_space_id": args.embedding_space_id,
            "reranker_model": args.reranker_model,
            "reranker_revision": args.reranker_revision or None,
            "corpus_sha256": fingerprint,
            "seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "report (company/year)",
        },
        "dataset": {
            "name": "FinQA",
            "license": "MIT",
            "commit": DATASET_COMMIT,
            "corpus_pages": len(corpus.documents),
            "corpus_chunks": sum(len(chunk_document(d)) for d in corpus.documents.values()),
            "companies": len(corpus.companies),
            "question_split": args.split,
            "test_split_unsealed": args.split == "test",
            "split_questions": split_questions,
            "selected_questions": len(cases),
            "selection": "first N rows (--limit)" if args.limit else "full split",
            "gold_labels": gold_label_audit(cases),
            **({"test_questions": len(cases)} if args.split == "test" else {}),
        },
        "embedding_model": args.embedding_model,
        "embedding_space_id": args.embedding_space_id,
        "quantization": args.quantization,
        "oversampling": args.oversampling,
    }
    if args.command in {"retrieval", "contexts", "answers"}:
        require_valid_gold(cases)
        report["dataset"]["evidence_mapping"] = evidence_audit(corpus, cases)
    if args.command == "ingest":
        report["ingest"] = await ingest(args, corpus)
    elif args.command == "retrieval":
        report["retrieval_config"] = {
            "top_k": 10,
            "rerank_candidates": args.rerank_candidates,
            "configs": args.configs,
            "scopes": args.scopes,
        }
        report["retrieval"] = await retrieval(args, corpus, cases)
    elif args.command == "contexts":
        from ledger_rag.evaluation.benchmarks.finqa_answers import retrieve_contexts

        report["contexts"] = await retrieve_contexts(args, corpus, cases)
        report["lineage"]["rerank_candidates"] = args.rerank_candidates
    elif args.command == "answers":
        from ledger_rag.evaluation.benchmarks.finqa_answers import answers
        from ledger_rag.evaluation.llm import ChatClient

        contexts = None
        if args.contexts:
            saved, source_hash = load_saved_contexts(
                args.contexts, fingerprint, args.collection, args.split
            )
            lineage = saved["lineage"]
            contexts = saved["contexts"]
            # Answer-only execution must retain the source's actual retrieval identity.
            for name in (
                "embedding_model",
                "embedding_revision",
                "embedding_space_id",
                "reranker_model",
                "reranker_revision",
                "rerank_candidates",
            ):
                if name in lineage:
                    report["lineage"][name] = lineage[name]
            report["lineage"]["contexts_source"] = {
                "path": str(args.contexts),
                "run_id": saved.get("run_id"),
                "git_sha": lineage["git_sha"],
                "embedding_space_id": lineage["embedding_space_id"],
                "reranker_revision": lineage["reranker_revision"],
                "report_sha256": source_hash,
            }
        self_hosted = args.llm_endpoint_kind == "self-hosted"
        stream = self_hosted if args.llm_stream is None else args.llm_stream
        client = ChatClient(
            args.llm_url,
            args.llm_key,
            concurrency=args.llm_concurrency,
            endpoint_kind=args.llm_endpoint_kind,
            stream=stream,
            request_timeout=args.llm_request_timeout,
            deadline_seconds=args.llm_deadline,
            requests_per_minute=args.llm_rpm,
        )
        server: dict[str, Any] | None = None
        try:
            if self_hosted:
                server = await client.probe(ready_timeout=args.llm_ready_timeout)
                check_served_models(args, server["models"])
            report["answers"] = await answers(args, corpus, cases, client, contexts)
        finally:
            await client.close()
        report["lineage"].update(
            {
                "llm_endpoint_host": args.llm_url.split("/")[2] if "//" in args.llm_url else "",
                "llm_endpoint_kind": args.llm_endpoint_kind,
                "llm_model_revision": args.llm_model_revision or None,
                "llm_server": server,
                "llm_stream": stream,
                "llm_models": report["answers"]["models"],
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "llm_seed": args.llm_seed,
                "llm_rpm_per_model": args.llm_rpm,
                "llm_concurrency": args.llm_concurrency,
                "llm_request_timeout": args.llm_request_timeout,
                "llm_deadline": args.llm_deadline,
                "prompt_sha256": report["answers"]["prompt_sha256"],
                "rerank_candidates": report["lineage"].get(
                    "rerank_candidates", args.rerank_candidates
                ),
            }
        )
    return report


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    report = asyncio.run(execute(args))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"answers", "retrieval"}}, indent=2
        )[:3000]
    )
    if "retrieval" in report:
        print(json.dumps({k: v["metrics"] for k, v in report["retrieval"]["arms"].items()})[:3000])
    if "answers" in report:
        from ledger_rag.evaluation.benchmarks.finqa_answers import brief

        print(brief(report["answers"]))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
