import asyncio
import random
from datetime import date

import httpx
import pytest

from ledger_rag.core.models import Chunk
from ledger_rag.evaluation.benchmarks.finqa_answers import (
    messages,
    paired_answer_differences,
    parse_mode,
    prompt_sha256,
    summarize_answers,
)
from ledger_rag.evaluation.evidence import citation_metrics, evidence_metrics, map_units
from ledger_rag.evaluation.llm import ChatClient, GenerationConfig
from ledger_rag.evaluation.scoring import gold_precision, parse_structured, score
from ledger_rag.evaluation.uncertainty import (
    cluster_bootstrap,
    latency,
    paired_cluster_difference,
    risk_coverage,
)


def _score(text: str, gold: float | str, stated: str = "") -> tuple[bool, bool, bool]:
    result = score(parse_structured(text), gold, stated)
    return result.exact, result.tolerant, result.percent_scale_diagnostic


def test_parse_uses_last_structured_lines_not_first_number() -> None:
    text = (
        "Revenue was 120 in 2020 and 100 in 2019.\nANSWER: 20%\nCONFIDENCE: 85\nCITATIONS: C2, C1"
    )
    parsed = parse_structured(text)
    assert parsed.value == 20 and parsed.percent
    assert parsed.confidence == pytest.approx(0.85)
    assert parsed.citations == (1, 2)
    assert parse_structured("Draft\nANSWER: 1\nfinal\nANSWER: 2").value == 2
    assert parse_structured("The answer is 20%").format_error == "missing"
    assert parse_structured("ANSWER: 120 or 100").format_error == "multiple_numbers"
    assert parse_structured("**ANSWER:** $1,234.5 million").value == 1234.5
    assert parse_structured("ANSWER: (3.2)%").value == -3.2
    assert parse_structured("ANSWER: \u22123.2%").value == -3.2
    assert parse_structured("ANSWER: INSUFFICIENT").abstained
    assert parse_structured("ANSWER: Yes.").text == "yes"


@pytest.mark.parametrize(
    ("text", "gold", "stated", "expected"),
    [
        ("ANSWER: 12.6%", 0.12574, "12.6%", (True, True, True)),
        ("ANSWER: 12.57%", 0.12574, "12.6%", (True, True, True)),
        ("ANSWER: 12.5%", 0.12574, "12.6%", (False, True, True)),
        # Bare 12.6 is not 0.126: percent scale needs an explicit % sign.
        ("ANSWER: 12.6", 0.12574, "12.6%", (False, False, True)),
        ("ANSWER: 0.126", 0.12574, "12.6%", (True, True, True)),
        ("ANSWER: 9119", 9119.0, "9119", (True, True, True)),
        ("ANSWER: 9120", 9119.0, "9119", (False, True, True)),
        ("ANSWER: 9300", 9119.0, "9119", (False, False, False)),
        ("ANSWER: -13.3%", -0.13327, "-13.3%", (True, True, True)),
        ("ANSWER: 13.3%", -0.13327, "-13.3%", (False, False, False)),
        ("ANSWER: yes", "yes", "yes", (True, True, False)),
        ("ANSWER: no", "yes", "yes", (False, False, False)),
        ("ANSWER: INSUFFICIENT", 1.0, "1", (False, False, False)),
        ("no structured line, 9119", 9119.0, "9119", (False, False, False)),
    ],
)
def test_exact_and_tolerant_are_separate(
    text: str, gold: float | str, stated: str, expected: tuple[bool, bool, bool]
) -> None:
    assert _score(text, gold, stated) == expected


def test_gold_precision_from_stated_answer() -> None:
    assert gold_precision(0.12574, "12.6%") == (3, "stated")
    assert gold_precision(9119.0, "9119") == (0, "stated")
    assert gold_precision(0.279, "27.9") == (3, "stated")  # unlabeled percent string
    assert gold_precision(5929.17, "73.7%") == (0, "fallback_3sf")
    assert gold_precision(0.00764, "") == (5, "fallback_3sf")


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        id=cid,
        document_id="d",
        tenant_id="t",
        title="t",
        text=text,
        parent_text=text,
        source="s",
        published_at=date(2020, 1, 1),
        ordinal=0,
        content_hash="h",
    )


def test_unit_mapping_whole_fragments_and_unmapped() -> None:
    chunks = [_chunk("a", "Alpha rose. Beta  fell."), _chunk("b", "Gamma held. | x | 1 |")]
    units = {
        "text_0": "Beta fell.",
        "text_1": "Beta fell. Gamma held.",
        "text_2": "Never ingested.",
        "table_1": "| x | 1 |",
    }
    mapping = map_units(units, chunks)
    assert mapping["text_0"].method == "whole" and mapping["text_0"].chunk_ids == {"a"}
    assert mapping["text_1"].method == "fragments"
    assert mapping["text_1"].satisfied_by(["a", "b"]) and not mapping["text_1"].satisfied_by(["a"])
    assert mapping["text_2"].method == "unmapped" and not mapping["text_2"].satisfied_by(["a", "b"])
    assert mapping["table_1"].chunk_ids == {"b"}
    metrics = evidence_metrics(["b", "a"], [mapping["text_0"], mapping["text_2"]], ks=(1, 2))
    # Unmapped gold stays in the denominator.
    assert metrics["evidence_recall@1"] == 0.0
    assert metrics["evidence_recall@2"] == 0.5
    assert metrics["all_evidence@2"] == 0.0
    assert metrics["evidence_precision@2"] == 0.5
    assert metrics["evidence_mrr@10"] == 0.5
    cites = citation_metrics(["a", "b", "a"], [mapping["text_0"], mapping["table_1"]])
    assert cites == {"citation_precision": 1.0, "citation_recall": 1.0}
    assert citation_metrics([], [mapping["text_0"]])["citation_precision"] is None


def test_cluster_bootstrap_widens_for_correlated_clusters() -> None:
    values = [1.0] * 50 + [0.0] * 50
    iid = cluster_bootstrap(values, [str(i) for i in range(100)], samples=500)
    clustered = cluster_bootstrap(values, ["a"] * 50 + ["b"] * 50, samples=500)
    assert iid["mean"] == clustered["mean"] == 0.5
    assert clustered["clusters"] == 2
    width = clustered["ci95"][1] - clustered["ci95"][0]
    assert width > iid["ci95"][1] - iid["ci95"][0]
    assert cluster_bootstrap([], []) == {"mean": None, "ci95": [None, None], "n": 0, "clusters": 0}
    assert cluster_bootstrap([0.5], ["x"], seed=3) == cluster_bootstrap([0.5], ["x"], seed=3)
    with pytest.raises(ValueError):
        cluster_bootstrap([1.0], [])


def test_paired_difference_and_risk_coverage() -> None:
    diff = paired_cluster_difference(
        {"a": 0.0, "b": 1.0, "c": 0.0}, {"a": 1.0, "b": 1.0, "d": 1.0}, {"a": "r", "b": "r"}
    )
    assert diff["delta"] == 0.5 and diff["n"] == 2
    curve = risk_coverage([0.9, 0.8, 0.1, 0.5], [True, True, False, False], points=(0.5, 1.0))
    assert curve["curve"] == [
        {"coverage": 0.5, "accuracy": 1.0},
        {"coverage": 1.0, "accuracy": 0.5},
    ]
    assert curve["aurc"] == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4, abs=1e-4)
    assert latency([1.0, 2.0, 3.0, 100.0])["p50_ms"] == 2.0


def test_answer_modes_prompt_and_summary() -> None:
    assert parse_mode("oracle") is None and parse_mode("retrieved@5") == 5
    with pytest.raises(ValueError):
        parse_mode("retrieved@11")
    built = messages("What?", [_chunk("a", "Alpha")])
    assert "ANSWER:" in built[0]["content"] and "[C1] t\nAlpha" in built[1]["content"]
    assert len(prompt_sha256()) == 64
    base = {
        "report": "R/2020",
        "answered": True,
        "abstained": False,
        "format_error": None,
        "error": None,
        "tolerant": True,
        "percent_scale_diagnostic": True,
        "gold_page_in_context": True,
        "context_evidence_recall": 1.0,
        "citation_precision": 1.0,
        "citation_recall": 1.0,
        "invalid_citations": 0,
        "confidence": 0.9,
        "top_retrieval_score": None,
        "usage": {"total_tokens": 10},
        "attempts": 1,
        "latency_ms": 5.0,
        "finish_reason": "stop",
    }
    records = [
        {**base, "id": "q1", "model": "m", "mode": "oracle", "exact": True},
        {**base, "id": "q1", "model": "m", "mode": "retrieved@3", "exact": False},
    ]
    summary = summarize_answers(records, seed=1, samples=50)
    assert summary["m/oracle"]["metrics"]["exact"]["mean"] == 1.0
    assert summary["m/oracle"]["token_usage"] == {"total_tokens": 10}
    deltas = paired_answer_differences(records, seed=1, samples=50)
    assert deltas["m: oracle - retrieved@3"]["delta"] == 1.0


def test_chat_client_retries_429_with_deadline_and_records_usage() -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(__import__("json").loads(request.content))
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ANSWER: 1"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    async def run() -> None:
        client = ChatClient("https://llm.test/v1", "k", rng=random.Random(0))
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client._rng = random.Random(0)
        config = GenerationConfig("m", temperature=0.0, reasoning_effort="low", seed=7)
        result = await client.complete(config, [{"role": "user", "content": "q"}])
        assert result.text == "ANSWER: 1" and result.attempts == 2
        assert result.usage["total_tokens"] == 5
        assert client.stats.retries == 1 and client.stats.status_counts == {"429": 1, "200": 1}
        assert calls[0]["reasoning_effort"] == "low" and calls[0]["seed"] == 7
        await client.close()

    asyncio.run(run())


def test_chat_client_does_not_retry_client_errors_and_bounds_concurrency() -> None:
    with pytest.raises(ValueError):
        ChatClient("https://llm.test/v1", "k", concurrency=9)
    with pytest.raises(ValueError):
        ChatClient("http://remote.test/v1", "k")

    async def run() -> None:
        client = ChatClient("https://llm.test/v1", "k")
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(400))
        )
        result = await client.complete(GenerationConfig("m"), [])
        assert result.error == "http_400" and result.attempts == 1
        assert client.stats.failures == 1
        await client.close()

    asyncio.run(run())
