"""Reproducible, query-clustered evaluation; no external benchmark content is bundled."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from ledger_rag.chunking import chunk_document
from ledger_rag.config import Settings, get_settings
from ledger_rag.models import Answer, Document, Principal, Profile, QueryRequest
from ledger_rag.retrieval import make_retriever
from ledger_rag.service import RagService

Metric = float | None
NUMBER = r"(?<![\w.])[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


class NumericalAssertion(BaseModel):
    """Match any captured number, in gold-specified units; not an arithmetic judge."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    expected: float
    absolute_tolerance: float = Field(default=0, ge=0)
    relative_tolerance: float = Field(default=0, ge=0)
    pattern: str = NUMBER

    @model_validator(mode="after")
    def valid_pattern(self) -> "NumericalAssertion":
        try:
            pattern = re.compile(self.pattern)
        except re.error as exc:
            raise ValueError("Invalid numerical assertion regex") from exc
        if pattern.groups > 1:
            raise ValueError("Numerical pattern must have at most one capture group")
        return self

    def matches(self, text: str) -> bool:
        for match in re.finditer(self.pattern, text):
            try:
                value = float(match.group(1 if match.re.groups else 0).replace(",", ""))
            except (ValueError, AttributeError):
                continue
            if math.isfinite(value) and math.isclose(
                value,
                self.expected,
                abs_tol=self.absolute_tolerance,
                rel_tol=self.relative_tolerance,
            ):
                return True
        return False


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=4000, pattern=r"\S")
    as_of: date | None = None
    relevant_document_ids: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] | None = None
    answer_contains: list[str] = Field(default_factory=list)
    numerical_assertions: list[NumericalAssertion] = Field(default_factory=list)
    should_abstain: bool = False
    slices: list[str] = Field(default_factory=list)
    split: str = "unspecified"

    @model_validator(mode="after")
    def valid_labels(self) -> "Case":
        if not self.should_abstain and not self.relevant_document_ids:
            raise ValueError("Answerable cases require positive document labels")
        if any(not value.strip() for value in self.answer_contains):
            raise ValueError("Empty answer assertions are vacuous")
        if self.relevant_chunk_ids == [] and not self.should_abstain:
            raise ValueError("Omit unavailable chunk labels rather than giving empty positives")
        return self


class Row(BaseModel):
    case_id: str
    repeat: int
    should_abstain: bool
    abstained: bool
    document_metrics: dict[str, Metric]
    chunk_metrics: dict[str, Metric] | None
    abstention_correct: bool
    answer_contains_pass: bool | None
    numerical_assertions_pass: bool | None
    labeled_answer_correct: bool | None
    citation_document_precision: Metric
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    retrieved_document_ids: list[str]
    retrieved_chunk_ids: list[str]
    cited_document_ids: list[str]
    slices: list[str]


def retrieval_metrics(
    retrieved: list[str], relevant: list[str], k: int | None = None
) -> dict[str, Metric]:
    """Binary relevance, deduplicate before cutoff. Undefined without positive gold."""
    unique = list(dict.fromkeys(retrieved))
    cutoff = k if k is not None else max(1, len(unique))
    if cutoff < 1:
        raise ValueError("k must be positive")
    unique = unique[:cutoff]
    gold = set(relevant)
    if not gold:
        return dict.fromkeys(("recall", "precision", "mrr", "ndcg"))
    found = len(set(unique) & gold)
    mrr = next((1 / rank for rank, item in enumerate(unique, 1) if item in gold), 0.0)
    dcg = sum(1 / math.log2(rank + 1) for rank, item in enumerate(unique, 1) if item in gold)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), cutoff) + 1))
    return {
        "recall": found / len(gold),
        "precision": found / cutoff,
        "mrr": mrr,
        "ndcg": dcg / ideal,
    }


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; empty samples have no percentile."""
    if not values or not 0 < fraction <= 1 or any(not math.isfinite(v) for v in values):
        raise ValueError("Require finite nonempty samples and a fraction in (0, 1]")
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def latency_summary(values: list[float]) -> dict[str, Metric]:
    return {
        label: percentile(values, fraction) if values else None
        for label, fraction in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
    }


def bootstrap_interval(
    values: list[float], *, seed: int = 42, samples: int = 1000
) -> dict[str, Metric | int]:
    """Percentile 95% CI of macro mean; callers first average repeats within case."""
    if samples < 1 or any(not math.isfinite(value) for value in values):
        raise ValueError("Require positive bootstrap samples and finite values")
    if not values:
        return {"low": None, "high": None, "cases": 0}
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples)]
    return {
        "low": percentile(means, 0.025),
        "high": percentile(means, 0.975),
        "cases": len(values),
    }


def question_leakage(cases: list[Case]) -> dict[str, Any]:
    groups: dict[str, list[Case]] = defaultdict(list)
    for case in cases:
        # Exact normalized duplicates only; no claim of detecting semantic paraphrases.
        groups[" ".join(case.question.casefold().split())].append(case)
    duplicates = [
        {
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "case_ids": [case.id for case in group],
            "splits": sorted({case.split for case in group}),
            "cross_split": len({case.split for case in group}) > 1,
        }
        for question, group in sorted(groups.items())
        if len(group) > 1
    ]
    return {
        "normalization": "Unicode casefold and whitespace collapse; not paraphrase detection",
        "unique_questions": len(groups),
        "duplicate_excess_cases": sum(len(group) - 1 for group in groups.values()),
        "cross_split_groups": sum(bool(group["cross_split"]) for group in duplicates),
        "groups": duplicates,
        "repeats_excluded": True,
    }


def score_answer(case: Case, answer: Answer) -> tuple[bool | None, bool | None, bool | None]:
    substrings = (
        all(needle.casefold() in answer.answer.casefold() for needle in case.answer_contains)
        if case.answer_contains
        else None
    )
    numbers = (
        all(assertion.matches(answer.answer) for assertion in case.numerical_assertions)
        if case.numerical_assertions
        else None
    )
    if case.should_abstain:
        correct: bool | None = answer.abstained
    elif substrings is None and numbers is None:
        correct = None if not answer.abstained else False
    else:
        correct = not answer.abstained and substrings is not False and numbers is not False
    return substrings, numbers, correct


def _row_metrics(row: Row) -> dict[str, Metric]:
    metrics: dict[str, Metric] = {
        "abstention_correct": float(row.abstention_correct),
        "abstention_sensitivity": float(row.abstained) if row.should_abstain else None,
        "abstention_specificity": float(not row.abstained) if not row.should_abstain else None,
        "coverage": float(not row.abstained),
        "risk_on_labeled_answers": (
            float(not row.labeled_answer_correct)
            if not row.abstained and row.labeled_answer_correct is not None
            else None
        ),
        "answer_contains_pass": (
            float(row.answer_contains_pass) if row.answer_contains_pass is not None else None
        ),
        "numerical_assertions_pass": (
            float(row.numerical_assertions_pass)
            if row.numerical_assertions_pass is not None
            else None
        ),
        "citation_document_precision": row.citation_document_precision,
    }
    for level, scores in (("document", row.document_metrics), ("chunk", row.chunk_metrics)):
        for name in ("recall", "precision", "mrr", "ndcg"):
            metrics[f"{level}_{name}_at_k"] = (
                scores[name] if scores is not None and not row.should_abstain else None
            )
    return metrics


def summarize(rows: list[Row], *, seed: int, bootstrap_samples: int) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for name, value in _row_metrics(row).items():
            if value is not None:
                grouped[name][row.case_id].append(value)
    metric_names = list(_row_metrics(rows[0])) if rows else []
    aggregate: dict[str, Metric] = {}
    intervals: dict[str, dict[str, Metric | int]] = {}
    for name in metric_names:
        values = [statistics.mean(values) for values in grouped[name].values()]
        aggregate[name] = statistics.mean(values) if values else None
        intervals[name] = bootstrap_interval(values, seed=seed, samples=bootstrap_samples)
    # Kept for consumers of the original smoke report.
    for old in ("recall", "precision", "mrr", "ndcg"):
        aggregate[old] = aggregate.get(f"document_{old}_at_k")
    tp = sum(row.should_abstain and row.abstained for row in rows)
    fn = sum(row.should_abstain and not row.abstained for row in rows)
    fp = sum(not row.should_abstain and row.abstained for row in rows)
    tn = sum(not row.should_abstain and not row.abstained for row in rows)
    answered = [row for row in rows if not row.abstained]
    labeled = [row for row in answered if row.labeled_answer_correct is not None]
    return {
        "aggregate": aggregate,
        "confidence_intervals_95": intervals,
        "abstention": {
            "positive_class": "should_abstain",
            "confusion": {
                "true_positive": tp,
                "false_negative": fn,
                "false_positive": fp,
                "true_negative": tn,
            },
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
        },
        "selective_prediction": {
            "coverage": len(answered) / len(rows) if rows else None,
            "risk_on_labeled_answers": (
                sum(row.labeled_answer_correct is False for row in labeled) / len(labeled)
                if labeled
                else None
            ),
            "answered_runs": len(answered),
            "labeled_answered_runs": len(labeled),
            "unlabeled_answered_runs": len(answered) - len(labeled),
            "warning": "Risk uses supplied lexical/numeric labels, not semantic correctness.",
        },
        "latency_ms": {
            f"{phase}_{label}": value
            for phase in ("retrieval", "generation", "total")
            for label, value in latency_summary(
                [getattr(row, f"{phase}_ms") for row in rows]
            ).items()
        },
        "cases": len({row.case_id for row in rows}),
        "runs": len(rows),
        "positive_retrieval_cases": len({row.case_id for row in rows if not row.should_abstain}),
    }


def _local_settings() -> Settings:
    # Explicit values override ambient remote/production environment: baseline never needs keys.
    return Settings(
        environment="development",
        auth_mode="demo",
        demo_token=SecretStr("evaluation-only"),
        provider="extractive",
        retrieval_backend="memory",
        seed_path=None,
        reranker_url="",
    )


def _source_digest() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.glob("*.py")):
        name = path.name.encode()
        content = path.read_bytes()
        # Length prefixes distinguish different file boundaries with equal concatenations.
        digest.update(len(name).to_bytes(8, "big") + name)
        digest.update(len(content).to_bytes(8, "big") + content)
    return digest.hexdigest()


async def evaluate(
    corpus: Path,
    cases_path: Path,
    repeats: int = 1,
    remote: bool = False,
    *,
    top_k: int = 5,
    seed: int = 42,
    bootstrap_samples: int = 1000,
    tenant: str | None = None,
    groups: list[str] | None = None,
    allow_index_write: bool = False,
    profile: Profile = Profile.CHAT,
) -> dict[str, Any]:
    if repeats < 1 or bootstrap_samples < 1:
        raise ValueError("Repeats and bootstrap samples must be positive")
    QueryRequest(question="validate configuration", top_k=top_k, profile=profile)
    if remote and not tenant:
        raise ValueError("Remote evaluation requires an explicit --tenant")
    settings = get_settings() if remote else _local_settings()
    if remote and settings.retrieval_backend == "qdrant" and not allow_index_write:
        raise ValueError("Corpus upsert mutates the remote index; require --allow-index-write")
    corpus_bytes, cases_bytes = await asyncio.gather(
        asyncio.to_thread(corpus.read_bytes), asyncio.to_thread(cases_path.read_bytes)
    )
    documents = [
        Document.model_validate_json(line) for line in corpus_bytes.splitlines() if line.strip()
    ]
    cases = [Case.model_validate_json(line) for line in cases_bytes.splitlines() if line.strip()]
    if not cases or not documents:
        raise ValueError("Evaluation cases and corpus must not be empty")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Case IDs must be unique; use --repeats for repeated trials")
    if len({(doc.tenant_id, doc.id) for doc in documents}) != len(documents):
        raise ValueError("Corpus document IDs must be unique within each tenant")
    principal = Principal(
        subject="evaluation", tenant_id=tenant or "demo", roles=["reader"], groups=groups or []
    )
    chunks = [chunk for doc in documents for chunk in chunk_document(doc)]
    known_documents = {doc.id for doc in documents}
    known_chunks = {chunk.id for chunk in chunks}
    if any(
        not set(case.relevant_document_ids) <= known_documents
        or not set(case.relevant_chunk_ids or []) <= known_chunks
        for case in cases
    ):
        raise ValueError("Gold labels must reference corpus documents/chunks")
    from ledger_rag.providers import make_provider

    store = make_retriever(settings)
    provider = make_provider(settings)
    service = RagService(settings, store, provider)
    effective_k = min(top_k, 3) if profile == Profile.VOICE else top_k
    rows: list[Row] = []
    try:
        await store.upsert(chunks)
        for repeat in range(repeats):
            for case in cases:
                start = time.perf_counter()
                query = QueryRequest(
                    question=case.question, as_of=case.as_of, top_k=top_k, profile=profile
                )
                async with asyncio.timeout(settings.query_deadline_seconds):
                    hits, elapsed = await service.retrieve(query, principal)
                    answer = await service.generate(query, hits, elapsed, case.id, principal)
                total_ms = (time.perf_counter() - start) * 1000
                identifiers = [hit.chunk.document_id for hit in hits]
                chunk_ids = [hit.chunk.id for hit in hits]
                cited = list(dict.fromkeys(c.document_id for c in answer.citations))
                substrings, numbers, correct = score_answer(case, answer)
                rows.append(
                    Row(
                        case_id=case.id,
                        repeat=repeat,
                        should_abstain=case.should_abstain,
                        abstained=answer.abstained,
                        document_metrics=retrieval_metrics(
                            identifiers, case.relevant_document_ids, effective_k
                        ),
                        chunk_metrics=(
                            retrieval_metrics(chunk_ids, case.relevant_chunk_ids, effective_k)
                            if case.relevant_chunk_ids is not None
                            else None
                        ),
                        abstention_correct=answer.abstained == case.should_abstain,
                        answer_contains_pass=substrings,
                        numerical_assertions_pass=numbers,
                        labeled_answer_correct=correct,
                        citation_document_precision=(
                            len(set(cited) & set(case.relevant_document_ids)) / len(cited)
                            if cited
                            else None
                        ),
                        retrieval_ms=elapsed,
                        generation_ms=answer.generation_ms,
                        total_ms=total_ms,
                        retrieved_document_ids=identifiers,
                        retrieved_chunk_ids=chunk_ids,
                        cited_document_ids=cited,
                        slices=case.slices,
                    )
                )
    finally:
        try:
            await store.close()
        finally:
            await provider.close()
    try:
        version = importlib.metadata.version("ledger-rag")
    except importlib.metadata.PackageNotFoundError:
        version = "uninstalled"
    summary = summarize(rows, seed=seed, bootstrap_samples=bootstrap_samples)
    return {
        "schema_version": "2",
        "dataset": str(cases_path),
        "backend": settings.retrieval_backend,
        "model": provider.model,
        "warning": (
            "Synthetic smoke data are not SOTA. Citation-document precision is relevance, "
            "NOT semantic faithfulness. Substring/numeric checks are limited label assertions. "
            "Remote evaluation calls the service directly: it does not test HTTP authentication."
        ),
        "manifest": {
            "created_at": datetime.now(UTC).isoformat(),
            "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
            "cases_sha256": hashlib.sha256(cases_bytes).hexdigest(),
            "package_version": version,
            "package_source_sha256": await asyncio.to_thread(_source_digest),
            "python_version": sys.version.split()[0],
            "provider": settings.provider,
            "model": provider.model,
            "model_revision": "Unreported; model identifiers may be mutable aliases",
            "config": {
                "remote": remote,
                "backend": settings.retrieval_backend,
                "embedding_model": (
                    settings.embedding_model if settings.retrieval_backend == "qdrant" else None
                ),
                "embedding_dimensions": (
                    settings.embedding_dimensions
                    if settings.retrieval_backend == "qdrant"
                    else None
                ),
                "top_k": top_k,
                "effective_k": effective_k,
                "profile": profile.value,
                "repeats": repeats,
                "seed": seed,
                "bootstrap_samples": bootstrap_samples,
                "max_tool_rounds": settings.max_tool_rounds,
                "query_deadline_seconds": settings.query_deadline_seconds,
                "llm_timeout_seconds": settings.llm_timeout_seconds,
                "reranker_enabled": bool(settings.reranker_url),
                "reranker_timeout_seconds": settings.reranker_timeout_seconds,
                "tenant": principal.tenant_id,
                "groups": sorted(principal.groups),
                "index_write": remote and settings.retrieval_backend == "qdrant",
            },
            "privacy": "Allowlisted configuration only; no credentials or endpoint URLs.",
            "execution": "Sequential queries; fail fast on runtime error, no failed-run censoring.",
        },
        **summary,
        "leakage": question_leakage(cases),
        "slices": {
            label: summarize(
                [row for row in rows if label in row.slices],
                seed=seed,
                bootstrap_samples=bootstrap_samples,
            )
            for label in sorted({label for row in rows for label in row.slices})
        },
        "results": [row.model_dump(mode="json") for row in rows],
    }


def check_thresholds(result: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    """Minimum metric / maximum latency gates; undefined metrics fail explicitly."""
    failures: list[str] = []
    if set(thresholds) - {"minimum", "maximum", "maximum_latency_ms", "slices"}:
        raise ValueError("Threshold sections: minimum, maximum, maximum_latency_ms, slices")
    for section, source, minimum in (
        ("minimum", result["aggregate"], True),
        ("maximum", result["aggregate"], False),
        ("maximum_latency_ms", result["latency_ms"], False),
    ):
        bounds = thresholds.get(section, {})
        if not isinstance(bounds, dict):
            raise ValueError(f"{section} must be an object")
        for name, bound in bounds.items():
            if (
                not isinstance(bound, (int, float))
                or isinstance(bound, bool)
                or not math.isfinite(bound)
                or bound < 0
                or (section != "maximum_latency_ms" and bound > 1)
            ):
                raise ValueError("Thresholds must be finite; metric minima must be in [0, 1]")
            if name not in source:
                raise ValueError(f"Unknown gate metric: {name}")
            value = source[name]
            if value is None or (value < bound if minimum else value > bound):
                failures.append(f"{section}.{name}")
    slices = thresholds.get("slices", {})
    if not isinstance(slices, dict):
        raise ValueError("slices must be an object")
    for label, gates in slices.items():
        if label not in result.get("slices", {}):
            failures.append(f"slices.{label}.missing")
        elif not isinstance(gates, dict) or "slices" in gates:
            raise ValueError("Slice thresholds must be non-nested gate objects")
        else:
            failures.extend(
                f"slices.{label}.{failure}"
                for failure in check_thresholds(result["slices"][label], gates)
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible retrieval/answer evaluation")
    parser.add_argument("--corpus", type=Path, default=Path("data/examples.jsonl"))
    parser.add_argument("--cases", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--profile", type=Profile, choices=list(Profile), default=Profile.CHAT)
    parser.add_argument("--remote", action="store_true", help="Configured endpoints; costs apply")
    parser.add_argument("--tenant", help="Explicit authorization scope; mandatory for --remote")
    parser.add_argument("--groups", nargs="*", default=[], help="Explicit authorized group IDs")
    parser.add_argument(
        "--allow-index-write", action="store_true", help="Permit Qdrant corpus upsert"
    )
    parser.add_argument("--smoke", action="store_true", help="Gate bundled synthetic smoke labels")
    parser.add_argument(
        "--thresholds", type=Path, help="JSON metric minima/maxima, latency maxima, slice gates"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = asyncio.run(
            evaluate(
                args.corpus,
                args.cases,
                args.repeats,
                args.remote,
                top_k=args.top_k,
                seed=args.seed,
                bootstrap_samples=args.bootstrap_samples,
                tenant=args.tenant,
                groups=args.groups,
                allow_index_write=args.allow_index_write,
                profile=args.profile,
            )
        )
        thresholds: dict[str, Any] = {}
        if args.smoke:
            thresholds = {
                "minimum": {
                    "document_recall_at_k": 1,
                    "abstention_correct": 1,
                    "answer_contains_pass": 1,
                }
            }
        if args.thresholds:
            thresholds = json.loads(args.thresholds.read_text())
            if not isinstance(thresholds, dict):
                raise ValueError("Threshold JSON must be an object")
        failures = check_thresholds(result, thresholds)
        result["gates"] = {"thresholds": thresholds, "failures": failures, "passed": not failures}
        serialized = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n")
        print(serialized)
    except (ValueError, OSError, RuntimeError, TimeoutError, httpx.HTTPError) as exc:
        # Settings validation errors can include secret input; never print exception content.
        parser.exit(2, f"Evaluation configuration/data failure ({type(exc).__name__}).\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
