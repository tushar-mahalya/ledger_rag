"""ASGI middleware and startup accounting for the vLLM server in ``compute/llm.py``.

vLLM's own ``--api-key`` guards only ``/v1``-style prefixes: ``/invocations``,
``/tokenize``, ``/metrics`` and other routes stay open, and the generation routes
among them would let anyone spend the GPU. ``BearerAuth`` is loaded with
``vllm serve --middleware llm_middleware.BearerAuth`` and:

- leaves only ``GET /health`` unauthenticated (readiness);
- requires ``Authorization: Bearer <key>`` everywhere else, compared in constant time
  against a SHA-256 digest, so the raw key never enters vLLM's environment;
- serves only an allowlist of routes (chat/completions, models, tokenize, version,
  Prometheus ``/metrics`` and ``/ledger/startup``) and returns 404 for the rest;
- requires ``Content-Length`` on POST and caps request bodies;
- writes one JSON access-log line per request (method, path, status, duration);
  headers, query strings and bodies are never logged.

``StartupTimings`` turns vLLM's startup log lines into a cold-start breakdown
(weights load, torch.compile, CUDA graphs, KV cache size) served at
``/ledger/startup``. This module is stdlib-only so it is testable without vLLM.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sys
import time
from collections.abc import Awaitable, Callable, MutableMapping
from pathlib import Path
from typing import Any

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

KEY_DIGEST_ENV = "LEDGER_LLM_SERVE_KEY_SHA256"
STARTUP_REPORT_ENV = "LEDGER_STARTUP_REPORT"
MAX_BODY_BYTES = 2 * 1024 * 1024
OPEN_ROUTES = frozenset({("GET", "/health")})
ROUTES = frozenset(
    {
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/completions"),
        ("GET", "/v1/models"),
        ("POST", "/tokenize"),
        ("GET", "/version"),
        ("GET", "/metrics"),
        ("GET", "/ledger/startup"),
    }
)


def log_event(event: str, **fields: Any) -> None:
    """One JSON object per line on stdout (Modal captures it as structured logs)."""
    record = {"ts": round(time.time(), 3), "event": event, **fields}
    sys.stdout.write(json.dumps(record, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def key_digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


async def _json(send: Send, status: int, body: dict[str, Any]) -> None:
    payload = json.dumps(body).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload})


class BearerAuth:
    """Pure ASGI middleware; vLLM instantiates it with the app as the only argument."""

    def __init__(self, app: ASGIApp, digest: str | None = None) -> None:
        digest = digest or os.environ.get(KEY_DIGEST_ENV, "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"{KEY_DIGEST_ENV} must hold the API key's SHA-256 hex digest")
        self.app = app
        self._digest = bytes.fromhex(digest)
        self._startup = Path(os.environ.get(STARTUP_REPORT_ENV, "/run/ledger/startup.json"))

    def _authorized(self, scope: Scope) -> bool:
        for name, value in scope.get("headers") or []:
            if name.lower() == b"authorization":
                scheme, _, token = value.decode("latin-1").partition(" ")
                if scheme.lower() != "bearer" or not token:
                    return False
                presented = hashlib.sha256(token.strip().encode()).digest()
                return hmac.compare_digest(presented, self._digest)
        return False

    @staticmethod
    def _content_length(scope: Scope) -> int | None:
        for name, value in scope.get("headers") or []:
            if name.lower() == b"content-length":
                try:
                    return int(value)
                except ValueError:
                    return None
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            await self.app(scope, receive, send)  # lifespan
            return
        started = time.perf_counter()
        method = str(scope.get("method", ""))
        path = str(scope.get("path", ""))
        status = 0

        async def recording_send(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            route = (method, path)
            length = self._content_length(scope) if method == "POST" else 0
            if route in OPEN_ROUTES:
                await self.app(scope, receive, recording_send)
            elif not self._authorized(scope):
                await _json(recording_send, 401, {"error": "unauthorized"})
            elif route not in ROUTES:
                await _json(recording_send, 404, {"error": "not found"})
            elif length is None:
                await _json(recording_send, 411, {"error": "content-length required"})
            elif length > MAX_BODY_BYTES:
                await _json(recording_send, 413, {"error": "request body too large"})
            elif path == "/ledger/startup":
                await _json(recording_send, 200, self._startup_report())
            else:
                await self.app(scope, receive, recording_send)
        finally:
            if path != "/health":
                log_event(
                    "http_request",
                    method=method,
                    path=path,
                    status=status,
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                )

    def _startup_report(self) -> dict[str, Any]:
        try:
            report: dict[str, Any] = json.loads(self._startup.read_text())
        except (OSError, ValueError):
            return {"error": "startup report unavailable"}
        return report


# vLLM v0.32 startup log lines (vllm/model_executor/model_loader, v1/worker,
# compilation, v1/engine/core.py, v1/core/kv_cache_utils.py, uvicorn).
_PATTERNS: dict[str, re.Pattern[str]] = {
    "weights_load_seconds": re.compile(r"Loading weights took ([\d.]+) seconds"),
    "model_load": re.compile(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds"),
    "dynamo_transform_seconds": re.compile(r"Dynamo bytecode transform time: ([\d.]+) s"),
    "compile_cache_load_seconds": re.compile(r"compile range .* from the cache, took ([\d.]+) s"),
    "compile_seconds": re.compile(r"torch\.compile took ([\d.]+) s in total"),
    "profiling_seconds": re.compile(r"Initial profiling/warmup run took ([\d.]+) s"),
    "cuda_graphs": re.compile(r"Graph capturing finished in ([\d.]+) secs, took ([\d.]+) GiB"),
    "init_engine_seconds": re.compile(
        r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s"
    ),
    "kv_cache": re.compile(
        r"KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per "
        r"request: ([\d.]+)x"
    ),
    "app_ready": re.compile(r"Application startup complete"),
}


# Phases that repeat per compiled component ("[backbone]", "[eagle_head]" for the MTP
# drafter) or per loaded checkpoint; the report keeps each part and their sum.
_SUMMED = frozenset(
    {
        "weights_load_seconds",
        "dynamo_transform_seconds",
        "compile_cache_load_seconds",
        "compile_seconds",
        "profiling_seconds",
    }
)
_COMPONENT = re.compile(r"\[[\w.]+:\d+\] \[([\w-]+)\] ")


class StartupTimings:
    """Cold-start breakdown parsed from vLLM's log stream.

    Wall-clock epochs let a client split its observed cold start into Modal
    scheduling plus image pull (until ``container_started``) and in-container work;
    ``events`` holds the epoch of the latest line of each phase.
    """

    def __init__(self, *, container_started: float, process_started: float) -> None:
        self.report: dict[str, Any] = {
            "container_started": round(container_started, 3),
            "process_started": round(process_started, 3),
            "events": {},
        }

    def observe(self, line: str, now: float | None = None) -> bool:
        """Record a matching log line; return True when the report changed."""
        now = time.time() if now is None else now
        events: dict[str, float] = self.report["events"]
        for name, pattern in _PATTERNS.items():
            match = pattern.search(line)
            if not match:
                continue
            groups = match.groups()
            if name in _SUMMED:
                parts: dict[str, float] = self.report.setdefault(f"{name}_parts", {})
                component = _COMPONENT.search(line)
                label = component.group(1) if component else str(len(parts))
                parts[label if label not in parts else f"{label}_{len(parts)}"] = float(groups[0])
                self.report[name] = round(sum(parts.values()), 3)
                events[name] = round(now, 3)
                return True
            if name in events:
                continue
            events[name] = round(now, 3)
            if name == "model_load":
                self.report["model_memory_gib"] = float(groups[0])
                self.report["model_load_seconds"] = float(groups[1])
            elif name == "cuda_graphs":
                self.report["cuda_graph_seconds"] = float(groups[0])
                self.report["cuda_graph_gib"] = float(groups[1])
            elif name == "kv_cache":
                self.report["kv_cache_tokens"] = int(groups[0].replace(",", ""))
                self.report["kv_cache_max_concurrency"] = float(groups[2])
            elif name == "app_ready":
                self.report["ready_seconds_after_process"] = round(
                    now - self.report["process_started"], 1
                )
            else:
                self.report[name] = float(groups[0])
            return True
        return False

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.report))
        temporary.replace(path)


_LEVEL = re.compile(r"^\(?(?:[\w-]+ pid=\d+\)\s*)?(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b")


def structured_line(line: str) -> str:
    """Wrap a vLLM text log line as JSON (JSON lines pass through unchanged)."""
    text = line.rstrip("\n")
    if text.startswith("{") and text.endswith("}"):
        return text
    match = _LEVEL.search(text)
    record = {
        "ts": round(time.time(), 3),
        "event": "vllm_log",
        "level": match.group(1) if match else "INFO",
        "message": text,
    }
    return json.dumps(record, separators=(",", ":"))
