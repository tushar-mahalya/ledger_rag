"""Endpoint-agnostic generation: self-hosted servers, arms, reasoning and sampling."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from ledger_rag.evaluation.benchmarks.finqa import (
    Case,
    Corpus,
    build_parser,
    check_served_models,
    main,
    page_document,
    validate_args,
)
from ledger_rag.evaluation.benchmarks.finqa_answers import (
    _answer_one,
    generation_configs,
    parse_arms,
    stratified_sample,
    stratum,
    summarize_answers,
)
from ledger_rag.evaluation.llm import ChatClient, Completion, GenerationConfig
from ledger_rag.evaluation.scoring import parse_structured
from ledger_rag.generation.reasoning import message_reasoning, split_reasoning

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def _args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": 2048,
        "llm_seed": None,
        "llm_arms": "",
        "llm_arms_file": None,
        "llm_models": ["m"],
        "llm_model_revision": "",
    }
    return argparse.Namespace(**(values | overrides))


@pytest.mark.parametrize(
    ("text", "answer", "reasoning"),
    [
        ("ANSWER: 1", "ANSWER: 1", ""),
        ("<think>ANSWER: 9</think>\nANSWER: 1", "\nANSWER: 1", "<think>ANSWER: 9</think>"),
        ("plan ANSWER: 9\n</think>\n\nANSWER: 1", "\n\nANSWER: 1", "plan ANSWER: 9\n</think>"),
        (
            "ANSWER: 1\n<think>never closed ANSWER: 9",
            "ANSWER: 1\n",
            "<think>never closed ANSWER: 9",
        ),
        ("<THINK>a</THINK>b", "b", "<THINK>a</THINK>"),
    ],
)
def test_split_reasoning_handles_closed_bare_and_unterminated_blocks(
    text: str, answer: str, reasoning: str
) -> None:
    assert split_reasoning(text) == (answer, reasoning)


def test_reasoning_fields_and_parser_never_score_leaked_thinking() -> None:
    assert message_reasoning({"reasoning_content": "r", "content": "c"}) == "r"
    assert message_reasoning({"reasoning": "r2"}) == "r2"
    assert message_reasoning({"content": "c", "reasoning_content": None}) == ""
    leaked = parse_structured("<think>ANSWER: 99\nCITATIONS: C9</think>\nANSWER: 5\nCITATIONS: C1")
    assert leaked.value == 5 and leaked.citations == (1,) and leaked.think_leak
    only_thinking = parse_structured("work...\nANSWER: 3\n</think>")
    assert only_thinking.format_error == "missing" and only_thinking.think_leak
    assert not parse_structured("ANSWER: 5").think_leak


def test_generation_config_extra_body_is_validated_merged_and_recorded() -> None:
    config = GenerationConfig("qwen", name="qwen:nothink", top_k=20, extra_body=NO_THINK)
    body = config.body([{"role": "user", "content": "q"}])
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["top_k"] == 20 and body["temperature"] == 0.0 and "seed" not in body
    manifest = config.manifest()
    assert manifest["label"] == "qwen:nothink" and manifest["extra_body"] == NO_THINK
    assert GenerationConfig("m", reasoning_effort="low").label == "m@low"
    for bad in ({"temperature": 1}, {"stream": True}, {"model": "x"}, {"seed": 1}):
        with pytest.raises(ValueError, match="may not set"):
            GenerationConfig("m", extra_body=bad)
    with pytest.raises(ValueError, match="JSON object"):
        GenerationConfig("m", extra_body=["x"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="JSON-serializable"):
        GenerationConfig("m", extra_body={"x": object()})
    with pytest.raises(ValueError, match="top_k"):
        GenerationConfig("m", top_k=1.5)  # type: ignore[arg-type]


def test_concurrency_and_pacing_depend_on_endpoint_kind() -> None:
    async def run() -> None:
        client = ChatClient(
            "https://llm.test/v1", "k", concurrency=256, endpoint_kind="self-hosted"
        )
        assert client.endpoint_kind == "self-hosted"
        await client.close()

    asyncio.run(run())
    with pytest.raises(ValueError, match="between 1 and 8"):
        ChatClient("https://llm.test/v1", "k", concurrency=9)
    with pytest.raises(ValueError, match="between 1 and 256"):
        ChatClient("https://llm.test/v1", "k", concurrency=257, endpoint_kind="self-hosted")
    with pytest.raises(ValueError, match="not paced"):
        ChatClient("https://llm.test/v1", "k", endpoint_kind="self-hosted", requests_per_minute=60)
    with pytest.raises(ValueError, match="endpoint_kind"):
        ChatClient("https://llm.test/v1", "k", endpoint_kind="cloud")


def _sse(*events: dict[str, Any]) -> bytes:
    lines = [f"data: {json.dumps(event)}\n\n" for event in events]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def test_streaming_separates_reasoning_and_records_usage_and_ttft() -> None:
    bodies: list[dict[str, Any]] = []
    stream = _sse(
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"reasoning_content": "think ANSWER: 9"}}]},
        {"choices": [{"delta": {"content": "ANSWER: 4"}}]},
        {"choices": [{"delta": {"content": "2"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=stream, headers={"content-type": "text/event-stream"})

    async def run() -> Completion:
        client = ChatClient("https://llm.test/v1", "k", endpoint_kind="self-hosted", stream=True)
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        config = GenerationConfig("qwen", extra_body=NO_THINK)
        try:
            return await client.complete(config, [{"role": "user", "content": "q"}])
        finally:
            await client.close()

    result = asyncio.run(run())
    assert result.text == "ANSWER: 42" and result.finish_reason == "stop"
    assert result.reasoning_chars == len("think ANSWER: 9")
    assert result.usage == {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}
    assert result.ttft_ms is not None and result.service_ms is not None
    assert bodies[0]["stream"] is True and bodies[0]["stream_options"] == {"include_usage": True}
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_non_streaming_reasoning_content_is_not_answer_text() -> None:
    payload = {
        "choices": [
            {
                "message": {"content": "ANSWER: 1", "reasoning_content": "secret plan"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"completion_tokens_details": {"reasoning_tokens": 3}, "total_tokens": 9},
    }

    async def run() -> Completion:
        client = ChatClient("https://llm.test/v1", "k")
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
        )
        try:
            return await client.complete(GenerationConfig("m"), [])
        finally:
            await client.close()

    result = asyncio.run(run())
    assert result.text == "ANSWER: 1" and result.reasoning_chars == len("secret plan")
    assert result.usage == {"total_tokens": 9, "reasoning_tokens": 3}
    assert result.ttft_ms is None


def test_deadline_stops_a_hung_request_without_retrying() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, json={})

    async def run() -> tuple[Completion, int]:
        client = ChatClient("https://llm.test/v1", "k", deadline_seconds=0.1)
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            return await client.complete(GenerationConfig("m"), []), client.stats.requests
        finally:
            await client.close()

    result, requests = asyncio.run(run())
    assert result.error == "deadline" and result.attempts == 1 and requests == 1


def test_probe_waits_for_cold_start_and_fails_fast_on_auth() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.32.0"})
        if len(calls) < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"data": [{"id": "m@abc"}, {"id": "m"}]})

    async def probe(transport: httpx.MockTransport, ready: float = 5) -> dict[str, Any]:
        client = ChatClient("https://llm.test/v1", "k", endpoint_kind="self-hosted")
        client._client = httpx.AsyncClient(transport=transport)
        try:
            return await client.probe(ready_timeout=ready, poll_seconds=0.01)
        finally:
            await client.close()

    server = asyncio.run(probe(httpx.MockTransport(handler)))
    assert server["models"] == ["m", "m@abc"] and server["version"] == "0.32.0"
    assert server["probe_attempts"] == 3 and calls[-1] == "/version"
    with pytest.raises(RuntimeError, match="http_401"):
        asyncio.run(probe(httpx.MockTransport(lambda _: httpx.Response(401))))
    with pytest.raises(TimeoutError, match="http_503"):
        asyncio.run(probe(httpx.MockTransport(lambda _: httpx.Response(503)), ready=0.05))


def test_arms_take_cli_defaults_and_reject_bad_specs(tmp_path: Path) -> None:
    arms = [
        {"label": "q:nothink", "model": "q", "extra_body": NO_THINK},
        {"label": "q:think-low", "model": "q", "max_tokens": 8192, "temperature": None},
    ]
    nothink, think = parse_arms(arms, _args(llm_seed=3))
    assert nothink.temperature == 0.0 and nothink.max_tokens == 2048 and nothink.seed == 3
    assert think.temperature is None and think.max_tokens == 8192
    assert "temperature" not in think.body([])
    path = tmp_path / "arms.json"
    path.write_text(json.dumps(arms))
    assert [c.label for c in generation_configs(_args(llm_arms_file=path))] == [
        "q:nothink",
        "q:think-low",
    ]
    assert [c.label for c in generation_configs(_args(llm_models=["a", "a:low"]))] == [
        "a",
        "a@low",
    ]
    bad_specs: list[tuple[object, str]] = [
        ([], "non-empty"),
        ([{"label": "x", "model": "q", "stream": True}], "Unknown"),
        ([{"label": "bad label", "model": "q"}], "label"),
        ([{"label": "x"}], "needs a model"),
        ([{"label": "x", "model": "q"}, {"label": "x", "model": "r"}], "Duplicate"),
        ([{"label": "x", "model": "q", "extra_body": {"seed": 1}}], "may not set"),
        ([{"label": "x", "model": "q", "temperature": 3}], "temperature"),
    ]
    for specs, message in bad_specs:
        with pytest.raises(ValueError, match=message):
            parse_arms(specs, _args())
    with pytest.raises(ValueError, match="either"):
        generation_configs(_args(llm_arms=json.dumps(arms), llm_arms_file=path))
    with pytest.raises(ValueError, match="valid JSON"):
        generation_configs(_args(llm_arms="[{"))
    with pytest.raises(ValueError, match="Duplicate"):
        generation_configs(_args(llm_models=["a", "a"]))


def _case(index: int, keys: list[str], program: str) -> Case:
    return Case(f"q{index}", "?", f"d{index}", "ACME", [], 1.0, program, gold_keys=keys)


def test_stratified_sample_is_proportional_seeded_and_ordered() -> None:
    assert stratum(_case(0, ["table_1", "text_2"], "subtract(1, 2), divide(#0, 2)")) == "both/2"
    assert stratum(_case(0, ["table_1"], "a(1), b(2), c(3), d(4)")) == "table/3+"
    assert stratum(_case(0, ["text_0"], "divide(1, 2)")) == "text/1"
    cases = [_case(i, ["table_1"], "divide(1, 2)") for i in range(60)]
    cases += [_case(60 + i, ["text_1"], "add(1, 2), divide(#0, 3)") for i in range(30)]
    cases += [_case(90 + i, ["table_1", "text_1"], "divide(1, 2)") for i in range(10)]
    sample = stratified_sample(cases, 20, seed=13)
    counts = {name: sum(stratum(c) == name for c in sample) for name in {stratum(c) for c in cases}}
    assert counts == {"table/1": 12, "text/2": 6, "both/1": 2}
    assert sample == stratified_sample(cases, 20, seed=13)
    assert sample != stratified_sample(cases, 20, seed=14)
    order = [cases.index(case) for case in sample]
    assert order == sorted(order)
    assert stratified_sample(cases, 500, seed=1) == cases
    # Largest remainder: 7 * (60, 30, 10) / 100 = 4.2, 2.1, 0.7 -> 4, 2, 1.
    small = stratified_sample(cases, 7, seed=13)
    assert sorted(stratum(c) for c in small).count("both/1") == 1


class _FakeClient:
    def __init__(self, completion: Completion) -> None:
        self.completion = completion

    async def complete(
        self, config: GenerationConfig, messages: list[dict[str, str]]
    ) -> Completion:
        return self.completion


def _corpus_and_case() -> tuple[Corpus, Case]:
    example = {
        "filename": "ACME/2020/page_4.pdf",
        "pre_text": ["revenue grew ."],
        "post_text": [],
        "table": [["", "2020"], ["net sales", "$ 120"]],
    }
    document, units = page_document(example)
    corpus = Corpus(documents={document.id: document}, units={document.id: units})
    return corpus, Case("q", "?", document.id, "ACME", [], 1.0, "")


@pytest.mark.parametrize(
    ("text", "finish", "reasoning_chars", "expected"),
    [
        ("ANSWER: 1\nCITATIONS: C1", "stop", 0, None),
        ("working it out", "length", 0, "truncated"),
        ("", "stop", 40, "reasoning_only"),
        ("<think>ANSWER: 1", "stop", 0, "reasoning_only"),
        ("I am unsure", "stop", 0, "missing"),
    ],
)
def test_answer_records_classify_truncation_and_reasoning_only(
    text: str, finish: str, reasoning_chars: int, expected: str | None
) -> None:
    corpus, case = _corpus_and_case()
    completion = Completion(
        text,
        finish,
        {"total_tokens": 4},
        1,
        5.0,
        service_ms=4.0,
        reasoning_chars=reasoning_chars,
        ttft_ms=1.0,
    )
    record = asyncio.run(
        _answer_one(
            _FakeClient(completion),  # type: ignore[arg-type]
            GenerationConfig("m", name="m:arm"),
            "oracle",
            case,
            corpus,
            [],
        )
    )
    assert record["format_error"] == expected and record["model"] == "m:arm"
    assert record["reasoning_chars"] == reasoning_chars and record["ttft_ms"] == 1.0
    assert record["exact"] is (expected is None)
    summary = summarize_answers([record], seed=1, samples=10)["m:arm/oracle"]
    assert summary["metrics"]["truncation_rate"]["mean"] == float(finish == "length")
    assert summary["mean_tokens"] == {"total_tokens": 4.0}
    assert summary["ttft"]["p50_ms"] == 1.0
    if expected:
        assert summary["format_errors"] == {expected: 1}


def _cli(*extra: str) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(["answers", *extra])
    validate_args(parser, args)
    return args


def test_cli_self_hosted_options_validate_and_check_served_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arms = json.dumps([{"label": "q:nothink", "model": "q", "extra_body": NO_THINK}])
    sha = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
    args = _cli(
        "--llm-endpoint-kind",
        "self-hosted",
        "--llm-concurrency",
        "64",
        "--llm-arms",
        arms,
        "--llm-model-revision",
        sha,
        "--answer-sample-strategy",
        "stratified",
    )
    assert args.llm_stream is None and args.llm_ready_timeout == 1200
    check_served_models(args, ["q", f"q@{sha}"])
    with pytest.raises(ValueError, match="does not advertise"):
        check_served_models(args, ["q"])
    with pytest.raises(ValueError, match="does not serve"):
        check_served_models(args, ["other"])
    for extra in (
        ["--llm-concurrency", "9"],
        ["--llm-endpoint-kind", "self-hosted", "--llm-concurrency", "257"],
        ["--llm-endpoint-kind", "self-hosted", "--llm-rpm", "10"],
        ["--llm-model-revision", "main"],
        ["--llm-arms", "[{"],
        ["--llm-deadline", "0"],
    ):
        monkeypatch.setattr(sys, "argv", ["finqa", "answers", *extra])
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 2
