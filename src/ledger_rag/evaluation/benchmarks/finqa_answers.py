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

Generation arms come from ``--llm-models`` (``model[:reasoning_effort]``) or from
JSON arm specs (``--llm-arms`` / ``--llm-arms-file``) that also carry sampling
parameters and a server-specific ``extra_body``, for example vLLM's
``{"chat_template_kwargs": {"enable_thinking": false}}``. Every arm is recorded.
"""

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ledger_rag.core.models import Chunk, QueryRequest
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
from ledger_rag.generation.reasoning import split_reasoning

PROMPT_VERSION = "finqa-structured-v2"
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
- Most questions need arithmetic (differences, sums, ratios, averages, percent changes)
  over figures in the context: do the calculation rather than abstaining.
- Only if a figure the calculation needs is absent from the context: ANSWER: INSUFFICIENT"""
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


ARM_FIELDS = frozenset(
    {
        "label",
        "model",
        "reasoning_effort",
        "temperature",
        "max_tokens",
        "seed",
        "top_p",
        "top_k",
        "presence_penalty",
        "extra_body",
    }
)
_ARM_LABEL = re.compile(r"[A-Za-z0-9._:@+=-]{1,80}")


def parse_arms(specs: object, args: argparse.Namespace) -> list[GenerationConfig]:
    """JSON arm specs: a list of objects with a unique ``label`` and a ``model``.

    Omitted ``temperature``, ``max_tokens`` and ``seed`` fall back to the CLI
    values; an explicit ``null`` omits the parameter so the server default applies.
    """
    if not isinstance(specs, list) or not specs:
        raise ValueError("LLM arms must be a non-empty JSON list")
    configs: list[GenerationConfig] = []
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("Each LLM arm must be a JSON object")
        unknown = sorted(set(spec) - ARM_FIELDS)
        if unknown:
            raise ValueError(f"Unknown LLM arm fields: {', '.join(unknown)}")
        label, model = spec.get("label"), spec.get("model")
        if not isinstance(label, str) or not _ARM_LABEL.fullmatch(label):
            raise ValueError("Each LLM arm needs a label of 1-80 [A-Za-z0-9._:@+=-] chars")
        if not isinstance(model, str) or not model:
            raise ValueError(f"LLM arm {label!r} needs a model")
        try:
            configs.append(
                GenerationConfig(
                    model=model,
                    name=label,
                    temperature=spec.get("temperature", args.temperature),
                    max_tokens=int(spec.get("max_tokens", args.max_tokens)),
                    reasoning_effort=spec.get("reasoning_effort"),
                    seed=spec.get("seed", args.llm_seed),
                    top_p=spec.get("top_p"),
                    top_k=spec.get("top_k"),
                    presence_penalty=spec.get("presence_penalty"),
                    extra_body=spec.get("extra_body") or {},
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"LLM arm {label!r}: {exc}") from None
    labels = Counter(config.label for config in configs)
    duplicates = sorted(label for label, count in labels.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate LLM arm labels: {', '.join(duplicates)}")
    return configs


def generation_configs(args: argparse.Namespace) -> list[GenerationConfig]:
    """Arms from ``--llm-arms``/``--llm-arms-file`` if given, else ``--llm-models``."""
    raw = getattr(args, "llm_arms", None)
    path = getattr(args, "llm_arms_file", None)
    if raw and path:
        raise ValueError("Use either --llm-arms or --llm-arms-file, not both")
    if path:
        raw = Path(path).read_text()
    if raw:
        try:
            specs = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM arms are not valid JSON: {exc.msg}") from None
        return parse_arms(specs, args)
    configs = [parse_model(spec, args) for spec in args.llm_models]
    labels = [config.label for config in configs]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate --llm-models entries")
    return configs


async def _answer_one(
    client: ChatClient,
    config: GenerationConfig,
    mode: str,
    case: Case,
    corpus: Corpus,
    hits: list[tuple[Chunk, float]],
) -> dict[str, Any]:
    evidence = corpus.evidence(case.document_id)
    k = parse_mode(mode)
    context = list(evidence.chunks) if k is None else [chunk for chunk, _ in hits[:k]]
    completion = await client.complete(config, messages(case.question, context))
    parsed = parse_structured(completion.text)
    format_error = parsed.format_error if not completion.error else None
    if format_error == "missing":
        answer_text, leaked = split_reasoning(completion.text)
        if completion.finish_reason == "length":
            format_error = "truncated"
        elif not answer_text.strip() and (leaked or completion.reasoning_chars):
            format_error = "reasoning_only"
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
        "format_error": format_error,
        "error": completion.error,
        "think_leak": parsed.think_leak,
        "reasoning_chars": completion.reasoning_chars,
        "confidence": parsed.confidence,
        "top_retrieval_score": hits[0][1] if hits and k is not None else None,
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
        "service_ms": completion.service_ms,
        "ttft_ms": completion.ttft_ms,
    }


def _row(record: dict[str, Any]) -> dict[str, float | None]:
    answered = record["answered"]
    return {
        "exact": float(record["exact"]),
        "tolerant": float(record["tolerant"]),
        "percent_scale_diagnostic": float(record["percent_scale_diagnostic"]),
        "abstention_rate": float(record["abstained"]),
        "format_failure_rate": float(bool(record["format_error"])),
        "truncation_rate": float(record["finish_reason"] == "length"),
        "think_leak_rate": float(bool(record.get("think_leak"))),
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
            "ttft": latency([r["ttft_ms"] for r in subset if r.get("ttft_ms") is not None]),
            "token_usage": usage,
            "mean_tokens": {
                key: round(value / len(subset), 1) for key, value in sorted(usage.items())
            },
            "mean_reasoning_chars": round(
                sum(r.get("reasoning_chars", 0) for r in subset) / len(subset), 1
            ),
            "format_errors": dict(
                sorted(Counter(str(r["format_error"]) for r in subset if r["format_error"]).items())
            ),
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


def stratum(case: Case) -> str:
    """Evidence source (table/text/both) x program steps (1, 2, 3+)."""
    kinds = {key.split("_", 1)[0] for key in case.gold_keys}
    source = "both" if {"table", "text"} <= kinds else "table" if "table" in kinds else "text"
    steps = len(re.findall(r"[a-z_]+\(", case.program))
    return f"{source}/{min(steps, 3)}{'+' if steps >= 3 else ''}"


def stratified_sample(cases: list[Case], size: int, seed: int) -> list[Case]:
    """Proportional allocation with largest-remainder rounding; seeded within strata.

    Returned in split order so runs over the same sample are comparable row by row.
    """
    if size >= len(cases):
        return list(cases)
    groups: dict[str, list[Case]] = {}
    for case in cases:
        groups.setdefault(stratum(case), []).append(case)
    quotas = {name: size * len(group) / len(cases) for name, group in groups.items()}
    counts = {name: int(quota) for name, quota in quotas.items()}
    by_remainder = sorted(groups, key=lambda name: (counts[name] - quotas[name], name))
    for name in by_remainder[: size - sum(counts.values())]:
        counts[name] += 1
    rng = random.Random(seed)
    chosen = {case.id for name in sorted(groups) for case in rng.sample(groups[name], counts[name])}
    return [case for case in cases if case.id in chosen]


def sample_cases(args: argparse.Namespace, cases: list[Case]) -> list[Case]:
    if getattr(args, "answer_sample_strategy", "random") == "stratified":
        return stratified_sample(cases, args.answer_samples, args.seed)
    return random.Random(args.seed).sample(cases, min(args.answer_samples, len(cases)))


def strata_counts(sample: list[Case], cases: list[Case]) -> dict[str, dict[str, int]]:
    population = Counter(stratum(case) for case in cases)
    sampled = Counter(stratum(case) for case in sample)
    return {
        name: {"population": population[name], "sampled": sampled[name]}
        for name in sorted(population)
    }


async def retrieve_contexts(
    args: argparse.Namespace, corpus: Corpus, cases: list[Case]
) -> dict[str, Any]:
    """Company-scoped answer retrieval for the sampled cases (the only GPU-bound step).

    Saved separately so LLM generation can run later without embedder/reranker.
    """
    sample = sample_cases(args, cases)
    ks = [k for k in (parse_mode(mode) for mode in args.answer_modes) if k is not None]
    top_k = max(ks, default=0)
    hits: dict[str, list[list[Any]]] = {case.id: [] for case in sample}
    retrieval_ms: list[float] = []
    if top_k:
        fusion, rerank, _ = arm_name(args.answer_retrieval)
        queries = [
            (QueryRequest(question=case.question, top_k=top_k), principal(case, "company", corpus))
            for case in sample
        ]
        outcomes = await search_all(args, fusion, rerank, queries, "answer-retrieval")
        for case, (found, elapsed) in zip(sample, outcomes, strict=True):
            hits[case.id] = [[h.chunk.document_id, h.chunk.id, h.score] for h in found]
            retrieval_ms.append(elapsed)
    return {
        "answer_retrieval": args.answer_retrieval,
        "scope": "company",
        "top_k": top_k,
        "sample_seed": args.seed,
        "samples": len(sample),
        "retrieval_latency": latency(retrieval_ms),
        "hits": hits,
    }


def _resolve(
    contexts: dict[str, Any], corpus: Corpus, sample: list[Case], ks: list[int], arm: str
) -> dict[str, list[tuple[Chunk, float]]]:
    """Validate saved contexts against this run and map chunk ids back to chunks."""
    if ks and (contexts["answer_retrieval"] != arm or contexts["top_k"] < max(ks)):
        raise ValueError("Saved contexts were built with a different retrieval arm or top-k")
    wanted = {case.id for case in sample}
    if not wanted <= set(contexts["hits"]):
        raise ValueError("Saved contexts do not cover the sampled cases")
    resolved: dict[str, list[tuple[Chunk, float]]] = {}
    for case_id, rows in contexts["hits"].items():
        if case_id not in wanted:
            continue
        pairs = []
        for document_id, chunk_id, value in rows:
            chunk = corpus.evidence(document_id).by_id.get(chunk_id)
            if chunk is None:
                raise ValueError(f"Saved context chunk {chunk_id} is not in the rebuilt corpus")
            pairs.append((chunk, float(value)))
        resolved[case_id] = pairs
    return resolved


async def answers(
    args: argparse.Namespace,
    corpus: Corpus,
    cases: list[Case],
    client: ChatClient,
    contexts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    configs = generation_configs(args)
    ks = [k for k in (parse_mode(mode) for mode in args.answer_modes) if k is not None]
    sample = sample_cases(args, cases)
    if contexts is None:
        contexts = await retrieve_contexts(args, corpus, cases)
    hits_by_case = _resolve(contexts, corpus, sample, ks, args.answer_retrieval)
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
        "models": [config.manifest() for config in configs],
        "modes": list(args.answer_modes),
        "answer_retrieval": args.answer_retrieval if ks else None,
        "samples": len(sample),
        "sample_seed": args.seed,
        "sample_strategy": getattr(args, "answer_sample_strategy", "random"),
        "strata": strata_counts(sample, cases),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(case.id for case in sample)).encode()
        ).hexdigest(),
        "retrieval_latency": contexts["retrieval_latency"],
        "wall_seconds": round(time.perf_counter() - started, 1),
        "llm_calls": {
            "completions": len(records),
            "http_requests": client.stats.requests,
            "retries": client.stats.retries,
            "failures": client.stats.failures,
            "status_counts": client.stats.status_counts,
            "error_samples": client.stats.error_samples,
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
