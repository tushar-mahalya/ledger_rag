"""FinQA answer accuracy with a structured final-answer contract.

Each (model, context mode, question) triple is one chat completion. The model sees
numbered context chunks and must finish with ``ANSWER:``, ``CONFIDENCE:`` and
``CITATIONS:`` lines; only those lines are scored (see ledger_rag.evaluation.scoring).
Context modes:

- ``retrieved@k``: top-k chunks from one company-scoped retrieval per question
  (default ``hybrid:rerank``), so ``retrieved@3`` is a prefix of ``retrieved@5``.
- ``oracle``: every chunk of the gold page, in page order.

This measures retrieval plus a single-shot reader. It deliberately bypasses the chat
service's calculator tool loop and citation validator so generation settings can be
fixed and recorded; it is not a measurement of the production chat path.
"""

import argparse
import asyncio
import hashlib
import json
import random
import sys
import time
from typing import Any

from ledger_rag.core.models import Chunk, Hit, QueryRequest
from ledger_rag.evaluation.benchmarks.finqa import (
    Case,
    Corpus,
    arm_name,
    principal,
    search_all,
)
from ledger_rag.evaluation.evidence import citation_metrics
from ledger_rag.evaluation.llm import ChatClient, GenerationConfig
from ledger_rag.evaluation.scoring import parse_structured, score
from ledger_rag.evaluation.uncertainty import (
    latency,
    paired_cluster_difference,
    risk_coverage,
    summarize_rows,
)

PROMPT_VERSION = "finqa-structured-v1"
SYSTEM_PROMPT = """You answer questions about excerpts from SEC annual reports (10-K).
Use only the numbered context. Think briefly (a few short lines at most), then end your
reply with exactly these three lines:
ANSWER: <final answer>
CONFIDENCE: <integer 0-100, your probability that ANSWER is correct>
CITATIONS: <comma-separated context ids you used, e.g. C1, C3; or NONE>

Rules for the ANSWER line:
- Give one value only: no words, currency symbols, thousands separators or scale words.
- Questions about a percentage, percent change, growth rate, share or portion "in percent":
  write a percentage with a % sign, e.g. ANSWER: 12.6%
- Other ratios: a plain decimal, e.g. ANSWER: 0.15
- Amounts: the number in the units used by the context, e.g. ANSWER: 9119
- Yes/no questions: ANSWER: yes or ANSWER: no
- Keep at least two decimal places unless the result is a whole number.
- If the context does not contain what is needed: ANSWER: INSUFFICIENT"""
USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}"


def prompt_sha256() -> str:
    return hashlib.sha256(f"{SYSTEM_PROMPT}\0{USER_TEMPLATE}".encode()).hexdigest()


def render_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[C{index}] {chunk.title}\n{chunk.text}" for index, chunk in enumerate(chunks, 1)
    )


def messages(question: str, chunks: list[Chunk]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(context=render_context(chunks), question=question),
        },
    ]


def parse_mode(mode: str) -> int | None:
    """``oracle`` -> None; ``retrieved@k`` -> k."""
    if mode == "oracle":
        return None
    name, _, k = mode.partition("@")
    if name != "retrieved" or not k.isdigit() or not 1 <= int(k) <= 10:
        raise ValueError(f"Unknown answer mode {mode!r}; use oracle or retrieved@k")
    return int(k)


def parse_model(spec: str, args: argparse.Namespace) -> GenerationConfig:
    """``model`` or ``model:reasoning_effort``."""
    model, _, effort = spec.partition(":")
    if not model:
        raise ValueError("Empty model name")
    return GenerationConfig(
        model=model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=effort or None,
        seed=args.llm_seed,
    )


async def _answer_one(
    client: ChatClient,
    config: GenerationConfig,
    mode: str,
    case: Case,
    corpus: Corpus,
    hits: list[Hit],
) -> dict[str, Any]:
    evidence = corpus.evidence(case.document_id)
    k = parse_mode(mode)
    context = list(evidence.chunks) if k is None else [hit.chunk for hit in hits[:k]]
    completion = await client.complete(config, messages(case.question, context))
    parsed = parse_structured(completion.text)
    result = score(parsed, case.answer, case.answer_text)
    gold = [evidence.units[key] for key in case.gold_keys]
    valid = [n for n in parsed.citations if 1 <= n <= len(context)]
    cited = [context[n - 1].id for n in valid]
    context_ids = [chunk.id for chunk in context]
    answered = not parsed.abstained and not completion.error
    return {
        "id": case.id,
        "report": case.report,
        "model": config.label,
        "mode": mode,
        "gold": case.answer,
        "gold_text": case.answer_text,
        "response": completion.text[-1500:],
        "parsed_answer": parsed.raw,
        "exact": result.exact,
        "tolerant": result.tolerant,
        "percent_scale_diagnostic": result.percent_scale_diagnostic,
        "gold_precision": result.precision,
        "gold_precision_source": result.precision_source,
        "abstained": parsed.abstained,
        "answered": answered,
        "format_error": parsed.format_error if not completion.error else None,
        "error": completion.error,
        "confidence": parsed.confidence,
        "top_retrieval_score": hits[0].score if hits and k is not None else None,
        "context": [f"{c.document_id}#{c.ordinal}" for c in context],
        "gold_page_in_context": any(c.document_id == case.document_id for c in context),
        "context_evidence_recall": (
            sum(unit.satisfied_by(context_ids) for unit in gold) / len(gold) if gold else None
        ),
        "citations": list(parsed.citations),
        "invalid_citations": len(parsed.citations) - len(valid),
        **citation_metrics(cited, gold),
        "finish_reason": completion.finish_reason,
        "usage": completion.usage,
        "attempts": completion.attempts,
        "latency_ms": completion.latency_ms,
    }


def _row(record: dict[str, Any]) -> dict[str, float | None]:
    answered = record["answered"]
    return {
        "exact": float(record["exact"]),
        "tolerant": float(record["tolerant"]),
        "percent_scale_diagnostic": float(record["percent_scale_diagnostic"]),
        "abstention_rate": float(record["abstained"]),
        "format_failure_rate": float(bool(record["format_error"])),
        "error_rate": float(bool(record["error"])),
        "selective_exact": float(record["exact"]) if answered else None,
        "gold_page_in_context": float(record["gold_page_in_context"]),
        "context_evidence_recall": record["context_evidence_recall"],
        "citation_precision": record["citation_precision"],
        "citation_recall": record["citation_recall"],
        "invalid_citation_rate": float(record["invalid_citations"] > 0),
    }


def summarize_answers(records: list[dict[str, Any]], *, seed: int, samples: int) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(f"{record['model']}/{record['mode']}", []).append(record)
    summary: dict[str, Any] = {}
    for label, subset in groups.items():
        answered = [r for r in subset if r["answered"]]
        usage: dict[str, int] = {}
        for record in subset:
            for key, value in record["usage"].items():
                usage[key] = usage.get(key, 0) + value
        selective: dict[str, Any] = {}
        confident = [r for r in answered if r["confidence"] is not None]
        if confident:
            selective["self_reported_confidence"] = risk_coverage(
                [r["confidence"] for r in confident], [r["exact"] for r in confident]
            )
        scored = [r for r in answered if r["top_retrieval_score"] is not None]
        if scored:
            selective["top_retrieval_score"] = risk_coverage(
                [r["top_retrieval_score"] for r in scored], [r["exact"] for r in scored]
            )
        summary[label] = {
            "metrics": summarize_rows(
                [_row(r) for r in subset],
                [r["report"] for r in subset],
                seed=seed,
                samples=samples,
            ),
            "coverage": round(len(answered) / len(subset), 4) if subset else None,
            "risk_coverage": selective,
            "latency_generation": latency([r["latency_ms"] for r in subset]),
            "token_usage": usage,
            "attempts": sum(r["attempts"] for r in subset),
            "finish_reasons": {
                str(reason): sum(r["finish_reason"] == reason for r in subset)
                for reason in sorted({str(r["finish_reason"]) for r in subset})
            },
        }
    return summary


def paired_answer_differences(
    records: list[dict[str, Any]], *, seed: int, samples: int
) -> dict[str, Any]:
    """Paired report-cluster deltas in exact accuracy across modes and models."""
    table: dict[tuple[str, str], dict[str, float]] = {}
    clusters: dict[str, str] = {}
    for record in records:
        table.setdefault((record["model"], record["mode"]), {})[record["id"]] = float(
            record["exact"]
        )
        clusters[record["id"]] = record["report"]
    models = sorted({model for model, _ in table})
    modes = sorted({mode for _, mode in table})
    out: dict[str, Any] = {}
    for model in models:
        for mode in modes:
            if mode != "oracle" and (model, "oracle") in table and (model, mode) in table:
                out[f"{model}: oracle - {mode}"] = paired_cluster_difference(
                    table[(model, mode)],
                    table[(model, "oracle")],
                    clusters,
                    seed=seed,
                    samples=samples,
                )
    for mode in modes:
        for first, second in zip(models, models[1:], strict=False):
            if (first, mode) in table and (second, mode) in table:
                out[f"{mode}: {second} - {first}"] = paired_cluster_difference(
                    table[(first, mode)],
                    table[(second, mode)],
                    clusters,
                    seed=seed,
                    samples=samples,
                )
    return out


async def answers(
    args: argparse.Namespace, corpus: Corpus, cases: list[Case], client: ChatClient
) -> dict[str, Any]:
    configs = [parse_model(spec, args) for spec in args.llm_models]
    ks = [k for k in (parse_mode(mode) for mode in args.answer_modes) if k is not None]
    sample = random.Random(args.seed).sample(cases, min(args.answer_samples, len(cases)))
    hits_by_case: dict[str, list[Hit]] = {case.id: [] for case in sample}
    retrieval_ms: list[float] = []
    if ks:
        fusion, rerank, _ = arm_name(args.answer_retrieval)
        queries = [
            (
                QueryRequest(question=case.question, top_k=max(ks)),
                principal(case, "company", corpus),
            )
            for case in sample
        ]
        outcomes = await search_all(args, fusion, rerank, queries, "answer-retrieval")
        for case, (hits, elapsed) in zip(sample, outcomes, strict=True):
            hits_by_case[case.id] = hits
            retrieval_ms.append(elapsed)
    started = time.perf_counter()
    done = 0

    async def run(config: GenerationConfig, mode: str, case: Case) -> dict[str, Any]:
        nonlocal done
        record = await _answer_one(client, config, mode, case, corpus, hits_by_case[case.id])
        done += 1
        if done % 50 == 0:
            print(f"answers: {done}/{total}", file=sys.stderr, flush=True)
        return record

    jobs = [
        (config, mode, case) for config in configs for mode in args.answer_modes for case in sample
    ]
    total = len(jobs)
    records = list(await asyncio.gather(*(run(*job) for job in jobs)))
    seed = args.seed
    samples = getattr(args, "bootstrap_samples", 2000)
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "models": [config.__dict__ | {"label": config.label} for config in configs],
        "modes": list(args.answer_modes),
        "answer_retrieval": args.answer_retrieval if ks else None,
        "samples": len(sample),
        "sample_seed": args.seed,
        "retrieval_latency": latency(retrieval_ms),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "llm_calls": {
            "completions": len(records),
            "http_requests": client.stats.requests,
            "retries": client.stats.retries,
            "failures": client.stats.failures,
            "status_counts": client.stats.status_counts,
        },
        "summary": summarize_answers(records, seed=seed, samples=samples),
        "paired_differences": paired_answer_differences(records, seed=seed, samples=samples),
        "records": records,
    }


def brief(result: dict[str, Any]) -> str:
    rows = {
        label: {
            name: value["mean"]
            for name, value in entry["metrics"].items()
            if name in {"exact", "tolerant", "abstention_rate"}
        }
        for label, entry in result["summary"].items()
    }
    return json.dumps(rows, indent=1)
