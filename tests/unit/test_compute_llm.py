import asyncio
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
KEY = "k" * 48


def _load(path: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


middleware = _load("compute/llm_middleware.py", "local_llm_middleware")


def _modal(path: str, name: str) -> ModuleType:
    pytest.importorskip("modal", reason="Modal app tests require the compute group")
    return _load(path, name)


def test_server_absolute_deadline_survives_restart() -> None:
    llm = _modal("compute/llm.py", "local_bounded_llm")
    assert llm.session_seconds_remaining("", 100) is None
    assert llm.session_seconds_remaining("150", 100) == 50
    for value in ("100", "99", "nan", "inf", "not-a-time"):
        with pytest.raises(ValueError):
            llm.session_seconds_remaining(value, 100)


def test_ephemeral_benchmark_url_cannot_redirect_the_bearer_key() -> None:
    bench = _modal("compute/llm_bench.py", "local_bounded_bench")
    bench.validate_serve_url("https://session--serve-dev.modal.run")
    for url in (
        "http://session--serve-dev.modal.run",
        "https://modal.run.attacker.test",
        "https://user@session--serve-dev.modal.run",
        "https://session--serve-dev.modal.run/v1",
        "https://session--serve-dev.modal.run?key=example",
    ):
        with pytest.raises(ValueError):
            bench.validate_serve_url(url)


def test_benchmark_checks_exact_revision_and_sdk_before_generation() -> None:
    bench = _modal("compute/llm_bench.py", "local_identity_bench")
    llm = _modal("compute/llm.py", "local_identity_serve")
    assert (bench.EXPECTED_MODEL, bench.EXPECTED_REVISION, bench.EXPECTED_VLLM) == (
        llm.MODEL,
        llm.MODEL_REVISION,
        llm.VLLM_VERSION,
    )
    served = [llm.MODEL, f"{llm.MODEL}@{llm.MODEL_REVISION}"]
    bench.validate_model_identity(served, llm.VLLM_VERSION)
    for models, version in (([llm.MODEL], llm.VLLM_VERSION), (served, "unknown")):
        with pytest.raises(ValueError, match="pinned"):
            bench.validate_model_identity(models, version)


@pytest.mark.asyncio
@pytest.mark.parametrize("valid_identity", [True, False])
async def test_first_cold_303_warms_and_checks_identity_before_chat(
    monkeypatch: pytest.MonkeyPatch, valid_identity: bool
) -> None:
    import httpx

    bench = _modal("compute/llm_bench.py", "local_cold_bench")
    probes = chats = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal probes
        if request.url.path == "/v1/models":
            probes += 1
            if probes == 1:
                return httpx.Response(303)
            served = [bench.EXPECTED_MODEL]
            if valid_identity:
                served.append(f"{bench.EXPECTED_MODEL}@{bench.EXPECTED_REVISION}")
            return httpx.Response(200, json={"data": [{"id": name} for name in served]})
        assert request.url.path == "/ledger/startup"
        return httpx.Response(
            200,
            json={
                "model": bench.EXPECTED_MODEL,
                "revision": bench.EXPECTED_REVISION,
                "vllm_version": bench.EXPECTED_VLLM,
                "events": {},
            },
        )

    async def no_sleep(seconds: float) -> None:
        pass

    async def first_chat(*args: Any) -> dict[str, int]:
        nonlocal chats
        chats += 1
        return {"status": 200}

    monkeypatch.setattr(bench.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(bench, "stream_chat", first_chat)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        if valid_identity:
            result = await bench.cold(client, "https://test.modal.run", bench.EXPECTED_MODEL)
            assert result["probe_attempts"] == 2 and result["probe_statuses"] == {"303": 1}
            assert chats == 1
        else:
            with pytest.raises(ValueError, match="pinned"):
                await bench.cold(client, "https://test.modal.run", bench.EXPECTED_MODEL)
            assert chats == 0


class Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        self.calls.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})


def _call(
    app: Any,
    method: str,
    path: str,
    headers: list[tuple[bytes, bytes]] | None = None,
    scope_type: str = "http",
) -> tuple[int | None, dict[str, Any], list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Any) -> None:
        sent.append(dict(message))

    scope = {"type": scope_type, "method": method, "path": path, "headers": headers or []}
    asyncio.run(app(scope, receive, send))
    status = next((m["status"] for m in sent if m["type"] == "http.response.start"), None)
    bodies = [m["body"] for m in sent if m["type"] == "http.response.body"]
    payload = json.loads(bodies[0]) if bodies and bodies[0] else {}
    return status, payload, sent


def _auth(key: str = KEY) -> list[tuple[bytes, bytes]]:
    return [(b"authorization", f"Bearer {key}".encode())]


@pytest.fixture
def guarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Recorder]:
    report = tmp_path / "startup.json"
    report.write_text(json.dumps({"events": {"app_ready": 1.0}}))
    monkeypatch.setenv(middleware.STARTUP_REPORT_ENV, str(report))
    inner = Recorder()
    return middleware.BearerAuth(inner, middleware.key_digest(KEY)), inner


def test_bearer_auth_requires_a_sha256_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(middleware.KEY_DIGEST_ENV, raising=False)
    with pytest.raises(RuntimeError, match="SHA-256"):
        middleware.BearerAuth(Recorder())
    monkeypatch.setenv(middleware.KEY_DIGEST_ENV, middleware.key_digest(KEY))
    assert middleware.BearerAuth(Recorder()).app is not None


def test_bearer_auth_leaves_only_health_open(guarded: tuple[Any, Recorder]) -> None:
    app, inner = guarded
    assert _call(app, "GET", "/health")[0] == 200
    for method, path in [
        ("GET", "/v1/models"),
        ("GET", "/metrics"),
        ("POST", "/invocations"),
        ("GET", "/ledger/startup"),
    ]:
        assert _call(app, method, path)[0] == 401
        assert _call(app, method, path, _auth("wrong-key"))[0] == 401
        assert _call(app, method, path, [(b"authorization", f"Basic {KEY}".encode())])[0] == 401
    assert inner.calls == ["/health"]


def test_bearer_auth_allowlists_routes_and_bounds_bodies(guarded: tuple[Any, Recorder]) -> None:
    app, inner = guarded
    assert _call(app, "GET", "/v1/models", _auth())[0] == 200
    assert _call(app, "GET", "/metrics", _auth())[0] == 200
    for method, path in [
        ("POST", "/invocations"),
        ("POST", "/v1/responses"),
        ("GET", "/docs"),
        ("POST", "/v1/models"),
        ("POST", "/pooling"),
    ]:
        assert _call(app, method, path, _auth())[0] == 404
    post = "/v1/chat/completions"
    assert _call(app, "POST", post, _auth())[0] == 411
    too_big = str(middleware.MAX_BODY_BYTES + 1).encode()
    assert _call(app, "POST", post, [*_auth(), (b"content-length", too_big)])[0] == 413
    assert _call(app, "POST", post, [*_auth(), (b"content-length", b"12")])[0] == 200
    assert inner.calls == ["/v1/models", "/metrics", post]


def test_bearer_auth_serves_startup_report_and_closes_websockets(
    guarded: tuple[Any, Recorder],
) -> None:
    app, inner = guarded
    status, payload, _ = _call(app, "GET", "/ledger/startup", _auth())
    assert status == 200 and payload == {"events": {"app_ready": 1.0}}
    _, _, sent = _call(app, "GET", "/v1/realtime", scope_type="websocket")
    assert sent == [{"type": "websocket.close", "code": 1008}]
    assert inner.calls == []


def test_access_log_omits_headers_and_keys(
    guarded: tuple[Any, Recorder], capsys: pytest.CaptureFixture[str]
) -> None:
    app, _ = guarded
    _call(app, "GET", "/health")
    _call(app, "GET", "/v1/models", _auth())
    _call(app, "GET", "/v1/models", _auth("leaked-secret-value"))
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(line["path"], line["status"]) for line in lines] == [
        ("/v1/models", 200),
        ("/v1/models", 401),
    ]
    assert all(
        set(line) == {"ts", "event", "method", "path", "status", "duration_ms"} for line in lines
    )
    assert KEY not in json.dumps(lines) and "leaked-secret-value" not in json.dumps(lines)


VLLM_STARTUP_LOG = [
    "(EngineCore pid=91) INFO 10-31 23:50:01 [default_loader.py:502] "
    "Loading weights took 41.27 seconds",
    "(EngineCore pid=91) INFO 10-31 23:50:01 [default_loader.py:502] "
    "Loading weights took 0.85 seconds",
    "(EngineCore pid=91) INFO 10-31 23:50:02 [gpu_model_runner.py:5150] "
    "Model loading took 29.4 GiB memory and 44.512345 seconds",
    "(EngineCore pid=91) INFO 10-31 23:50:20 [backends.py:1167] "
    "[backbone] Dynamo bytecode transform time: 14.02 s",
    "(EngineCore pid=91) INFO 10-31 23:50:25 [backends.py:297] [backbone] Directly load the "
    "compiled graph(s) for compile range (1, 8192) from the cache, took 4.513 s",
    "(EngineCore pid=91) INFO 10-31 23:51:00 [monitor.py:62] "
    "[backbone] torch.compile took 52.10 s in total",
    "(EngineCore pid=91) INFO 10-31 23:51:05 [monitor.py:91] "
    "[backbone] Initial profiling/warmup run took 3.21 s",
    "(EngineCore pid=91) INFO 10-31 23:51:05 [monitor.py:62] "
    "[eagle_head] torch.compile took 5.06 s in total",
    "(EngineCore pid=91) INFO 10-31 23:51:06 [kv_cache_utils.py:2328] GPU KV cache size: "
    "412,160 tokens, Maximum concurrency for 32,768 tokens per request: 12.58x",
    "(EngineCore pid=91) INFO 10-31 23:51:30 [gpu_model_runner.py:6582] "
    "Graph capturing finished in 24 secs, took 1.07 GiB",
    "(EngineCore pid=91) INFO 10-31 23:51:31 [core.py:396] init engine (profile, create kv "
    "cache, warmup model) took 88.40 s (compilation: 52.10 s)",
    "(APIServer pid=1) INFO:     Application startup complete.",
]


def test_startup_timings_parse_vllm_log_lines(tmp_path: Path) -> None:
    timings = middleware.StartupTimings(container_started=1000.0, process_started=1010.0)
    changed = [timings.observe(line, now=1010.0 + i) for i, line in enumerate(VLLM_STARTUP_LOG)]
    assert all(changed)
    assert not timings.observe("Model loading took 1.0 GiB memory and 1.0 seconds", now=2e3)
    assert not timings.observe("(APIServer pid=1) INFO unrelated", now=2000.0)
    report = timings.report
    assert report["weights_load_seconds"] == 42.12
    assert report["weights_load_seconds_parts"] == {"0": 41.27, "1": 0.85}
    assert (report["model_memory_gib"], report["model_load_seconds"]) == (29.4, 44.512345)
    assert report["dynamo_transform_seconds"] == 14.02
    assert report["compile_cache_load_seconds"] == 4.513
    assert report["compile_seconds"] == 57.16
    assert report["compile_seconds_parts"] == {"backbone": 52.10, "eagle_head": 5.06}
    assert report["events"]["compile_seconds"] == 1017.0
    assert report["profiling_seconds"] == 3.21
    assert (report["kv_cache_tokens"], report["kv_cache_max_concurrency"]) == (412160, 12.58)
    assert (report["cuda_graph_seconds"], report["cuda_graph_gib"]) == (24.0, 1.07)
    assert report["init_engine_seconds"] == 88.40
    assert report["ready_seconds_after_process"] == 11.0
    assert report["events"]["app_ready"] == 1021.0
    path = tmp_path / "run" / "startup.json"
    timings.write(path)
    assert json.loads(path.read_text()) == report
    assert not path.with_suffix(".tmp").exists()


def test_structured_line_wraps_text_and_passes_json_through() -> None:
    record = json.loads(middleware.structured_line(VLLM_STARTUP_LOG[0] + "\n"))
    assert record["event"] == "vllm_log" and record["level"] == "INFO"
    assert record["message"] == VLLM_STARTUP_LOG[0]
    warning = "(APIServer pid=1) WARNING 10-31 23:50:01 [api_server.py:1] slow"
    assert json.loads(middleware.structured_line(warning))["level"] == "WARNING"
    assert middleware.structured_line('{"event":"x"}\n') == '{"event":"x"}'


def test_serving_app_pins_model_and_follows_the_vllm_recipe() -> None:
    llm = _modal("compute/llm.py", "local_llm_serve")
    assert re.fullmatch(r"[0-9a-f]{40}", llm.MODEL_REVISION)
    assert llm.MODEL_REVISION.startswith("017b9c7af6b5")
    argv = llm.vllm_argv(3)
    joined = " ".join(argv)

    def value(flag: str) -> str:
        return str(argv[argv.index(flag) + 1])

    assert argv[:3] == ["vllm", "serve", llm.MODEL]
    assert value("--revision") == value("--tokenizer-revision") == llm.MODEL_REVISION
    assert f"{llm.MODEL} {llm.MODEL}@{llm.MODEL_REVISION}" in joined
    assert value("--reasoning-parser") == "qwen3"
    assert value("--tool-call-parser") == "qwen3_coder"
    assert value("--kv-cache-dtype") == "fp8"
    assert value("--max-model-len") == "32768"
    assert value("--gpu-memory-utilization") == "0.9"
    assert value("--middleware") == "llm_middleware.BearerAuth"
    for flag in ("--language-model-only", "--enable-auto-tool-choice", "--disable-fastapi-docs"):
        assert flag in argv
    assert "--api-key" not in argv  # the key never reaches vLLM's argv or environment
    assert json.loads(value("--speculative-config")) == {
        "method": "mtp",
        "num_speculative_tokens": 3,
    }
    assert "--speculative-config" not in llm.vllm_argv(0)
    with pytest.raises(ValueError):
        llm.vllm_argv(9)


def test_serving_app_scales_to_zero_on_a_single_h100() -> None:
    llm = _modal("compute/llm.py", "local_llm_serve_settings")
    source = (ROOT / "compute/llm.py").read_text()
    assert llm.GPU == "H100!"
    assert "min_containers=0" in source and "max_containers=1" in source
    assert 60 <= llm.SCALEDOWN_SECONDS <= 600
    assert llm.STARTUP_TIMEOUT_SECONDS >= 15 * 60
    assert llm.app.name == "ledger-llm-serve"


def test_bench_parses_spec_decode_counters_and_acceptance() -> None:
    bench = _modal("compute/llm_bench.py", "local_llm_bench")
    text = "\n".join(
        [
            "# HELP vllm:spec_decode_num_drafts_total Number of drafts.",
            "# TYPE vllm:spec_decode_num_drafts_total counter",
            'vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 100.0',
            'vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} 300.0',
            'vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 231.0',
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 90.0',
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="1"} 78.0',
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="2"} 63.0',
            'vllm:spec_decode_num_drafts_created{engine="0"} 1.7e9',
        ]
    )
    after = bench.parse_counters(text)
    assert after["drafts"] == 100 and after["accepted_tokens"] == 231
    assert after["per_position"] == {0: 90.0, 1: 78.0, 2: 63.0}
    before = {"drafts": 0.0, "draft_tokens": 0.0, "accepted_tokens": 0.0, "per_position": {}}
    result = bench.acceptance(before, after)
    assert result["acceptance_rate"] == 0.77
    assert result["mean_acceptance_length"] == 3.31
    assert result["per_position_acceptance"] == {"0": 0.9, "1": 0.78, "2": 0.63}
    assert bench.acceptance(after, after)["acceptance_rate"] is None


def test_bench_helpers_are_deterministic_and_unique() -> None:
    bench = _modal("compute/llm_bench.py", "local_llm_bench_helpers")
    assert bench.percentile([], 0.5) is None
    assert bench.percentile([3.0, 1.0, 2.0, 4.0], 0.5) == 2.0
    assert bench.percentile([3.0, 1.0, 2.0, 4.0], 0.95) == 4.0
    assert bench.context(7, 10) == bench.context(7, 10) != bench.context(8, 10)
    assert bench.messages("x") != bench.messages("x")
    with pytest.raises(ValueError, match="unknown phases"):
        asyncio.run(bench.run("https://example.test", KEY, ["cold", "warp"]))
