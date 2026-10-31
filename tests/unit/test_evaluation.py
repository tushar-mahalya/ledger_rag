import asyncio
import hashlib
import json
import math
from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from ledger_rag.core.config import Settings
from ledger_rag.core.models import Answer, Citation, Profile
from ledger_rag.evaluation.harness import (
    Case,
    NumericalAssertion,
    Row,
    bootstrap_interval,
    check_thresholds,
    evaluate,
    latency_summary,
    percentile,
    question_leakage,
    retrieval_metrics,
    score_answer,
    summarize,
)
from scripts.benchmark_http import LoadConfig, benchmark, query_once

ROOT = Path(__file__).resolve().parents[2]


def test_binary_metrics_known_ranks_and_duplicate_invariance() -> None:
    result = retrieval_metrics(["a", "a", "b", "b", "c"], ["b", "c", "c"], 3)
    assert result["recall"] == 1
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["mrr"] == 0.5
    assert result["ndcg"] == pytest.approx(
        (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    )
    assert result == retrieval_metrics(["a", "b", "c"], ["b", "c"], 3)
    assert retrieval_metrics(["a", "a", "b"], ["b"], 2)["mrr"] == 0.5


def test_cutoff_and_missing_results_do_not_shrink_ideal_ranking() -> None:
    assert retrieval_metrics(["a"], ["a", "b"], 3) == {
        "recall": 0.5,
        "precision": 1 / 3,
        "mrr": 1,
        "ndcg": 1 / (1 + 1 / math.log2(3)),
    }
    assert retrieval_metrics(["x", "a"], ["a"], 1) == dict.fromkeys(
        ["recall", "precision", "mrr", "ndcg"], 0.0
    )
    assert retrieval_metrics([], ["a"], 5) == dict.fromkeys(
        ["recall", "precision", "mrr", "ndcg"], 0.0
    )


def test_unanswerable_retrieval_is_undefined_not_perfect() -> None:
    assert all(value is None for value in retrieval_metrics([], []).values())
    assert all(value is None for value in retrieval_metrics(["a"], []).values())
    with pytest.raises(ValueError, match="positive"):
        retrieval_metrics([], [], 0)


@pytest.mark.parametrize("fraction,expected", [(0.5, 2), (0.95, 4), (0.99, 4), (1, 4)])
def test_percentile_is_nearest_rank(fraction: float, expected: float) -> None:
    assert percentile([4, 1, 2, 3], fraction) == expected
    assert percentile([8], fraction) == 8


@pytest.mark.parametrize(
    "values,fraction",
    [
        ([], 0.5),
        ([1], 0),
        ([1], 1.01),
        ([math.nan], 0.5),
        ([math.inf], 0.5),
    ],
)
def test_invalid_percentiles(values: list[float], fraction: float) -> None:
    with pytest.raises(ValueError):
        percentile(values, fraction)


def test_bootstrap_seed_degenerate_and_missing_samples() -> None:
    assert bootstrap_interval([0, 1], seed=7) == {"low": 0, "high": 1, "cases": 2}
    assert bootstrap_interval([0.3, 0.3], samples=10) == {
        "low": 0.3,
        "high": 0.3,
        "cases": 2,
    }
    assert bootstrap_interval([]) == {"low": None, "high": None, "cases": 0}
    values = [0.1, 0.2, 0.6, 0.9]
    assert bootstrap_interval(values, seed=12) == bootstrap_interval(values, seed=12)
    with pytest.raises(ValueError):
        bootstrap_interval(values, samples=0)
    with pytest.raises(ValueError):
        bootstrap_interval([math.nan])
    assert latency_summary([]) == dict.fromkeys(["p50", "p95", "p99"])


def make_row(
    case_id: str,
    *,
    gold_abstain: bool = False,
    abstained: bool = False,
    recall: float = 1,
    labeled_correct: bool | None = True,
    repeat: int = 0,
) -> Row:
    return Row(
        case_id=case_id,
        repeat=repeat,
        should_abstain=gold_abstain,
        abstained=abstained,
        document_metrics=dict.fromkeys(["recall", "precision", "mrr", "ndcg"], recall),
        chunk_metrics=None,
        abstention_correct=gold_abstain == abstained,
        answer_contains_pass=None,
        numerical_assertions_pass=None,
        labeled_answer_correct=labeled_correct,
        citation_document_precision=None,
        retrieval_ms=1,
        generation_ms=2,
        total_ms=3,
        retrieved_document_ids=[],
        retrieved_chunk_ids=[],
        cited_document_ids=[],
        slices=[],
    )


def test_positive_retrieval_excludes_unanswerable_and_optional_labels() -> None:
    rows = [
        make_row("answer", recall=0.5),
        make_row("missing", gold_abstain=True, abstained=True, recall=1),
    ]
    report = summarize(rows, seed=42, bootstrap_samples=20)
    assert report["aggregate"]["document_recall_at_k"] == 0.5
    assert report["aggregate"]["chunk_recall_at_k"] is None
    assert report["aggregate"]["answer_contains_pass"] is None
    assert report["aggregate"]["citation_document_precision"] is None
    assert report["positive_retrieval_cases"] == 1
    assert report["confidence_intervals_95"]["document_recall_at_k"]["cases"] == 1


def test_bootstrap_clusters_repeats_before_macro_aggregation() -> None:
    rows = [
        make_row("a", recall=0),
        make_row("a", recall=1, repeat=1),
        make_row("b", recall=1),
    ]
    report = summarize(rows, seed=42, bootstrap_samples=100)
    assert report["aggregate"]["recall"] == 0.75
    assert report["confidence_intervals_95"]["document_recall_at_k"] == {
        "low": 0.5,
        "high": 1,
        "cases": 2,
    }
    assert report["cases"] == 2
    assert report["runs"] == 3


def test_confusion_coverage_risk_and_unknown_label_denominator() -> None:
    rows = [
        make_row("tp", gold_abstain=True, abstained=True),
        make_row("fn", gold_abstain=True, labeled_correct=False),
        make_row("fp", abstained=True, labeled_correct=False),
        make_row("tn"),
        make_row("unknown", labeled_correct=None),
    ]
    report = summarize(rows, seed=1, bootstrap_samples=10)
    assert report["abstention"]["confusion"] == {
        "true_positive": 1,
        "false_negative": 1,
        "false_positive": 1,
        "true_negative": 2,
    }
    assert report["abstention"]["sensitivity"] == 0.5
    assert report["abstention"]["specificity"] == 2 / 3
    selective = report["selective_prediction"]
    assert selective["coverage"] == 3 / 5
    assert selective["risk_on_labeled_answers"] == 0.5
    assert selective["unlabeled_answered_runs"] == 1
    assert selective["labeled_answered_runs"] == 2
    empty = summarize([], seed=1, bootstrap_samples=10)
    assert empty["abstention"]["specificity"] is None
    assert empty["selective_prediction"]["coverage"] is None


def make_answer(text: str = "Revenue was 120 million [C1].", *, abstained: bool = False) -> Answer:
    return Answer(
        answer=text,
        citations=[]
        if abstained
        else [
            Citation(
                id="C1",
                document_id="a",
                source="https://example.invalid",
                title="Synthetic",
                published_at=date(2025, 1, 1),
                quote="120 million",
            )
        ],
        abstained=abstained,
        profile=Profile.CHAT,
        retrieval_ms=1,
        generation_ms=2,
        request_id="test",
        model="synthetic",
    )


def test_optional_labels_are_not_vacuous_successes() -> None:
    case = Case(id="a", question="Revenue?", relevant_document_ids=["a"])
    assert score_answer(case, make_answer()) == (None, None, None)
    case.answer_contains = ["revenue", "120"]
    assert score_answer(case, make_answer()) == (True, None, True)
    assert score_answer(case, make_answer(abstained=True)) == (True, None, False)
    case.answer_contains = ["150"]
    assert score_answer(case, make_answer()) == (False, None, False)
    negative = Case(id="negative", question="unknown", should_abstain=True)
    assert score_answer(negative, make_answer()) == (None, None, False)
    assert score_answer(negative, make_answer(abstained=True)) == (None, None, True)


def test_numerical_labels_units_tolerance_and_capture() -> None:
    exact = NumericalAssertion(expected=120, pattern=r"Revenue was ([\d,.]+) million")
    assert exact.matches("Revenue was 120.000 million [C1].")
    assert not exact.matches("Revenue was 120 billion [C1].")
    assert not exact.matches("Revenue was 12 million, debt 120 million")
    assert NumericalAssertion(expected=1200).matches("1,200")
    assert NumericalAssertion(expected=-2.5).matches("-2.5")
    assert NumericalAssertion(expected=1000).matches("1e3")
    assert NumericalAssertion(expected=100, absolute_tolerance=0.1).matches("100.05")
    assert not NumericalAssertion(expected=100, absolute_tolerance=0.1).matches("100.2")
    assert NumericalAssertion(expected=100, relative_tolerance=0.01).matches("100.5")
    assert NumericalAssertion(expected=0, absolute_tolerance=0.01).matches("0.009")
    assert not NumericalAssertion(expected=1, pattern=r"value=(\S+)").matches("value=nan")
    case = Case(
        id="a", question="Revenue?", relevant_document_ids=["a"], numerical_assertions=[exact]
    )
    assert score_answer(case, make_answer()) == (None, True, True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected": math.inf},
        {"expected": 1, "absolute_tolerance": -1},
        {"expected": 1, "pattern": "["},
        {"expected": 1, "pattern": "(a)(b)"},
    ],
)
def test_invalid_numeric_labels(kwargs: dict[str, Any]) -> None:
    with pytest.raises((ValueError, ValidationError)):
        NumericalAssertion.model_validate(kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"relevant_document_ids": []},
        {"relevant_document_ids": ["a"], "answer_contains": [" "]},
        {"relevant_document_ids": ["a"], "relevant_chunk_ids": []},
        {"relevant_document_ids": ["a"], "unknown_label": True},
    ],
)
def test_invalid_case_labels(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Case.model_validate({"id": "a", "question": "x", **kwargs})


def test_leakage_is_normalized_cross_split_and_no_question_text() -> None:
    cases = [
        Case(id="a", question="WHAT was revenue?", relevant_document_ids=["a"], split="train"),
        Case(id="b", question="what  was revenue? ", relevant_document_ids=["a"], split="test"),
        Case(id="c", question="What was debt?", relevant_document_ids=["a"], split="test"),
    ]
    report = question_leakage(cases)
    assert report["unique_questions"] == 2
    assert report["duplicate_excess_cases"] == 1
    assert report["cross_split_groups"] == 1
    assert report["groups"][0]["case_ids"] == ["a", "b"]
    assert "what was revenue" not in json.dumps(report)


def test_gate_bounds_missing_metrics_and_slices() -> None:
    report = summarize([make_row("a", recall=0.5)], seed=1, bootstrap_samples=10)
    report["slices"] = {"finance": report.copy()}
    assert check_thresholds(report, {"minimum": {"recall": 0.4}}) == []
    assert check_thresholds(
        report,
        {
            "minimum": {"chunk_recall_at_k": 0},
            "maximum_latency_ms": {"total_p95": 2},
            "slices": {"finance": {"minimum": {"recall": 0.9}}, "missing": {}},
        },
    ) == [
        "minimum.chunk_recall_at_k",
        "maximum_latency_ms.total_p95",
        "slices.finance.minimum.recall",
        "slices.missing.missing",
    ]
    assert check_thresholds(report, {"maximum": {"risk_on_labeled_answers": 0}}) == []
    invalid_thresholds: list[dict[str, Any]] = [
        {"minimum": {"unknown": 0.5}},
        {"minimum": {"recall": math.nan}},
        {"minimum": {"recall": True}},
        {"minimum": {"recall": 2}},
        {"typo": {}},
        {"slices": []},
    ]
    for thresholds in invalid_thresholds:
        with pytest.raises(ValueError):
            check_thresholds(report, thresholds)


@pytest.mark.asyncio
async def test_baseline_repeats_manifest_and_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEDGER_PROVIDER", "openai")
    monkeypatch.setenv("LEDGER_RETRIEVAL_BACKEND", "qdrant")
    monkeypatch.setenv("LEDGER_RERANKER_URL", "http://never-contact.invalid")
    monkeypatch.setenv("LEDGER_ENVIRONMENT", "production")
    result = await evaluate(ROOT / "data/examples.jsonl", ROOT / "data/eval.jsonl", repeats=2)
    assert result["cases"] == 4 and result["runs"] == 8
    assert result["positive_retrieval_cases"] == 3
    assert result["backend"] == "memory"
    assert result["aggregate"]["recall"] == 1
    assert result["abstention"]["specificity"] == 1
    assert result["abstention"]["sensitivity"] == 1
    assert result["confidence_intervals_95"]["document_recall_at_k"]["cases"] == 3
    assert (
        result["manifest"]["corpus_sha256"]
        == hashlib.sha256((ROOT / "data/examples.jsonl").read_bytes()).hexdigest()
    )
    manifest = json.dumps(result["manifest"])
    assert "evaluation-only" not in manifest
    assert "never-contact" not in manifest
    assert result["manifest"]["config"]["reranker_enabled"] is False
    assert result["leakage"]["duplicate_excess_cases"] == 0
    assert (
        check_thresholds(
            result,
            {
                "minimum": {
                    "recall": 1,
                    "abstention_correct": 1,
                    "answer_contains_pass": 1,
                }
            },
        )
        == []
    )
    negative = [row for row in result["results"] if row["should_abstain"]][0]
    assert all(value is None for value in negative["document_metrics"].values())
    assert negative["answer_contains_pass"] is None


@pytest.mark.asyncio
async def test_voice_effective_k_and_tenant_acl() -> None:
    result = await evaluate(
        ROOT / "data/examples.jsonl",
        ROOT / "data/eval.jsonl",
        tenant="nonexistent",
        profile=Profile.VOICE,
    )
    assert result["manifest"]["config"]["effective_k"] == 3
    assert result["aggregate"]["recall"] == 0
    assert all(row["abstained"] for row in result["results"])


@pytest.mark.asyncio
async def test_remote_requires_scope_and_index_write_before_any_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="tenant"):
        await evaluate(ROOT / "data/examples.jsonl", ROOT / "data/eval.jsonl", remote=True)
    settings = Settings(
        demo_token=SecretStr("test-secret"),
        retrieval_backend="qdrant",
        embedding_api_key=SecretStr("never-print-this"),
    )
    monkeypatch.setattr("ledger_rag.evaluation.harness.get_settings", lambda: settings)
    with pytest.raises(ValueError, match="allow-index-write"):
        await evaluate(
            ROOT / "data/examples.jsonl",
            ROOT / "data/eval.jsonl",
            remote=True,
            tenant="demo",
        )


@pytest.mark.asyncio
async def test_repeat_and_bootstrap_config_reject_zero() -> None:
    with pytest.raises(ValueError, match="positive"):
        await evaluate(ROOT / "data/examples.jsonl", ROOT / "data/eval.jsonl", repeats=0)
    with pytest.raises(ValueError, match="positive"):
        await evaluate(ROOT / "data/examples.jsonl", ROOT / "data/eval.jsonl", bootstrap_samples=0)


class TestStream(httpx.AsyncByteStream):
    __test__ = False

    def __init__(self, chunks: list[bytes], delay: float = 0) -> None:
        self.chunks = chunks
        self.delay = delay

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            await asyncio.sleep(self.delay)
            yield chunk


def sse_response(*, done: bool = True, error: bool = False) -> httpx.Response:
    answer = make_answer().model_dump_json()
    chunks = [
        b": heartbeat\n\n",
        b'event: evidence\ndata: {"sources": [], "retrieval_ms": 1}\n\n',
        f"event: answer\ndata: {answer}\n\n".encode(),
    ]
    if error:
        chunks.append(b'event: error\ndata: {"detail": "do not log body"}\n\n')
    elif done:
        chunks.append(b"event: done\ndata: {}\n\n")
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, stream=TestStream(chunks, 0.002)
    )


@pytest.mark.asyncio
async def test_http_fixed_concurrency_budgets_auth_warmup_and_no_secret_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LEDGER_BENCHMARK_TOKEN", "private-test-token")
    active = peak = count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak, count
        assert request.headers["authorization"] == "Bearer private-test-token"
        assert "groups" not in request.headers
        active += 1
        count += 1
        peak = max(peak, active)
        await asyncio.sleep(0.002)
        active -= 1
        return httpx.Response(200, json=make_answer().model_dump(mode="json"))

    result = await benchmark(
        LoadConfig(requests=10, concurrency=3, warmup=2),
        transport=httpx.MockTransport(handler),
    )
    assert result["attempted"] == result["succeeded"] == 10
    assert count == 12 and peak == 3
    assert result["success_rate"] == 1
    assert result["throughput_requests_per_second"] > 0
    assert result["errors"] == {}
    assert result["latency_ms"]["ttfe_successful_sse"]["p50"] is None
    assert result["latency_sample_counts"]["validated_answer_successful"] == 10
    assert "private-test-token" not in json.dumps(result)


@pytest.mark.asyncio
async def test_http_failures_are_counted_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEDGER_BENCHMARK_TOKEN", "test")
    statuses = iter([200, 429, 401, 500, 302])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(next(statuses), json={"invalid": True})

    result = await benchmark(
        LoadConfig(requests=5, warmup=0), transport=httpx.MockTransport(handler)
    )
    assert result["success_rate"] == 0
    assert result["errors"] == {
        "invalid_response": 1,
        "http_429": 1,
        "http_401": 1,
        "http_500": 1,
        "http_302": 1,
    }
    assert result["latency_ms"]["validated_answer_successful"]["p95"] is None
    assert result["latency_sample_counts"]["all_request_completion"] == 5


@pytest.mark.asyncio
async def test_sse_ttfe_is_not_validated_answer_latency() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: sse_response())
    ) as client:
        result = await query_once(client, LoadConfig(stream=True))
    assert result.success and result.error is None
    assert result.ttfe_ms is not None and result.validated_answer_ms is not None
    assert result.validated_answer_ms > result.ttfe_ms
    assert result.elapsed_ms >= result.validated_answer_ms


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "done,error,expected",
    [
        (False, False, "invalid_response"),
        (False, True, "sse_error"),
    ],
)
async def test_sse_incomplete_or_error_after_answer_is_not_success(
    done: bool,
    error: bool,
    expected: str,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: sse_response(done=done, error=error))
    ) as client:
        result = await query_once(client, LoadConfig(stream=True))
    assert not result.success and result.error == expected
    assert result.validated_answer_ms is not None


@pytest.mark.asyncio
async def test_sse_absolute_timeout_and_transport_failures() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: sse_response())
    ) as client:
        result = await query_once(client, LoadConfig(stream=True, timeout=0.001))
    assert not result.success and result.error == "timeout"

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sensitive endpoint must not be returned")

    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
        result = await query_once(client, LoadConfig())
    assert result.error == "transport_error"


@pytest.mark.asyncio
async def test_sse_chunked_utf8_crlf_bom_and_multiline_data() -> None:
    answer = make_answer("Révenue was 120 million [C1].").model_dump_json()
    wire = (
        '\ufeffevent: evidence\r\ndata: {"sources": [],\r\n'
        'data: "retrieval_ms": 1}\r\n\r\n'
        f"event: answer\r\ndata: {answer}\r\n\r\n"
        "event: done\r\ndata: {}\r\n\r\n"
    ).encode()
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=TestStream([bytes([byte]) for byte in wire]),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response)) as client:
        result = await query_once(client, LoadConfig(stream=True))
    assert result.success


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_response_budget_precedes_unterminated_line_buffering(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    monkeypatch.setattr("scripts.benchmark_http.MAX_RESPONSE_BYTES", 32)
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=TestStream([b"x" * 20, b"x" * 20]),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response)) as client:
        result = await query_once(client, LoadConfig(stream=stream))
    assert not result.success and result.error == "invalid_response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    [
        "event: done\ndata: {}\n\n",
        'event: evidence\ndata: {"sources": [], "retrieval_ms": true}\n\n',
        'event: evidence\ndata: {"sources": [], "retrieval_ms": -1}\n\n',
        'event: evidence\ndata: {"sources": [], "retrieval_ms": NaN}\n\n',
        "event: unknown\ndata: {}\n\n",
        "event: evidence\ndata: not-json\n\n",
    ],
)
async def test_sse_invalid_events_and_ordering(events: str) -> None:
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=TestStream([events.encode()]),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response)) as client:
        result = await query_once(client, LoadConfig(stream=True))
    assert not result.success and result.error == "invalid_response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"answer": " "},
        {"citations": []},
        {"abstained": True},
        {"profile": "voice"},
        {"retrieval_ms": -1},
        {"generation_ms": math.inf},
    ],
)
async def test_http_invalid_answer_contract(overrides: dict[str, Any]) -> None:
    payload = make_answer().model_dump(mode="json") | overrides
    # json.dumps explicitly permits Infinity to verify the client rejects nonfinite data.
    response = httpx.Response(200, content=json.dumps(payload).encode())
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response)) as client:
        result = await query_once(client, LoadConfig())
    assert not result.success and result.error == "invalid_response"


@pytest.mark.asyncio
async def test_duration_budget_drains_without_unbounded_arrivals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LEDGER_BENCHMARK_TOKEN", "test")

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.003)
        return httpx.Response(200, json=make_answer().model_dump(mode="json"))

    result = await benchmark(
        LoadConfig(requests=None, duration=0.01, concurrency=2, warmup=0),
        transport=httpx.MockTransport(handler),
    )
    assert 2 <= result["attempted"] < 100
    assert result["succeeded"] == result["attempted"]
    assert result["elapsed_seconds"] >= 0.01


@pytest.mark.parametrize(
    "kwargs",
    [
        {"url": "http://user:password@example.com"},
        {"url": "http://example.com?token=secret"},
        {"url": "file:///etc/passwd"},
        {"concurrency": 0},
        {"requests": 0},
        {"timeout": math.nan},
        {"warmup": -1},
        {"requests": None},
        {"duration": 1},
        {"requests": None, "duration": math.inf},
    ],
)
def test_http_invalid_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        LoadConfig(**kwargs).validate()


@pytest.mark.asyncio
async def test_http_bearer_is_environment_only_and_mandatory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LEDGER_BENCHMARK_TOKEN", raising=False)
    with pytest.raises(ValueError, match="bearer"):
        await benchmark(LoadConfig(requests=1))
    monkeypatch.setenv("LEDGER_BENCHMARK_TOKEN", "bad\r\nheader")
    with pytest.raises(ValueError, match="bearer"):
        await benchmark(LoadConfig(requests=1))
