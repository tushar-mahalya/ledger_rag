"""Bounded, deadline-aware OpenAI-compatible chat client for offline evaluation.

The production providers deliberately expose a narrow request surface. Benchmarks
need explicit, recorded generation settings (sampling, token cap, seed, reasoning
controls, server-specific ``extra_body``), token usage, and retry accounting, so
evaluation uses this small client instead. It never logs request bodies, keys or
response bodies.

Endpoint kinds:

- ``hosted``: a metered third-party API. Concurrency is capped at 8 and optional
  client-side pacing (``requests_per_minute``) keeps runs inside provider quotas.
- ``self-hosted``: an unthrottled server we operate (for example vLLM on Modal).
  Pacing is disabled, concurrency may reach 256 so the server can batch, and
  ``probe`` waits for readiness and reports the served model ids and version.

Streaming (SSE) keeps long generations alive behind proxies that cap the time to
the first response byte, and measures time to first token.
"""

import asyncio
import json
import random
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, get_args

import httpx

from ledger_rag.generation.reasoning import message_reasoning

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
EndpointKind = Literal["hosted", "self-hosted"]
ENDPOINT_KINDS: tuple[str, ...] = get_args(EndpointKind)
MAX_CONCURRENCY = {"hosted": 8, "self-hosted": 256}
# Request fields the client owns; arms set sampling through explicit fields instead.
RESERVED_BODY_KEYS = frozenset(
    {
        "model",
        "messages",
        "stream",
        "stream_options",
        "n",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "seed",
        "reasoning_effort",
    }
)


def validate_extra_body(extra: object) -> dict[str, Any]:
    """A JSON object merged into the request; it may not override owned fields."""
    if not isinstance(extra, Mapping):
        raise ValueError("extra_body must be a JSON object")
    reserved = sorted(RESERVED_BODY_KEYS & set(extra))
    if reserved:
        raise ValueError(f"extra_body may not set {', '.join(reserved)}")
    try:
        json.dumps(extra)
    except (TypeError, ValueError):
        raise ValueError("extra_body must be JSON-serializable") from None
    return dict(extra)


@dataclass(frozen=True)
class GenerationConfig:
    """One generation arm. ``None`` fields are omitted (server defaults apply)."""

    model: str
    temperature: float | None = 0.0
    max_tokens: int = 2048
    reasoning_effort: str | None = None
    seed: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty")
        for name in ("seed", "top_k"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                raise ValueError(f"{name} must be an integer")
        if self.reasoning_effort is not None and not isinstance(self.reasoning_effort, str):
            raise ValueError("reasoning_effort must be a string")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be within [0, 2]")
        if not 1 <= self.max_tokens <= 131072:
            raise ValueError("max_tokens must be within [1, 131072]")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be within (0, 1]")
        if self.top_k is not None and self.top_k < -1:
            raise ValueError("top_k must be -1 (disabled) or positive")
        if self.presence_penalty is not None and not -2 <= self.presence_penalty <= 2:
            raise ValueError("presence_penalty must be within [-2, 2]")
        validate_extra_body(self.extra_body)

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        return f"{self.model}@{self.reasoning_effort}" if self.reasoning_effort else self.model

    def body(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        for key in ("temperature", "top_p", "top_k", "presence_penalty", "reasoning_effort"):
            value = getattr(self, key)
            if value is not None:
                body[key] = value
        if self.seed is not None:
            body["seed"] = self.seed
        body.update(self.extra_body)
        return body

    def manifest(self) -> dict[str, Any]:
        return asdict(self) | {"label": self.label}


@dataclass
class Completion:
    text: str
    finish_reason: str | None
    usage: dict[str, int]
    attempts: int
    latency_ms: float
    error: str | None = None
    # Duration of the final HTTP attempt only (excludes pacing and retry waits).
    service_ms: float | None = None
    # Characters of server-separated reasoning (never parsed for answers).
    reasoning_chars: int = 0
    # Time to the first streamed content or reasoning token (streaming only).
    ttft_ms: float | None = None


@dataclass
class CallStats:
    requests: int = 0
    retries: int = 0
    failures: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    error_samples: list[str] = field(default_factory=list)


@dataclass
class _Reply:
    text: str
    reasoning: str
    finish_reason: str | None
    payload: dict[str, Any]
    ttft_ms: float | None = None


def _usage(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage") or {}
    out: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if isinstance(usage.get(key), int):
            out[key] = usage[key]
    details = usage.get("completion_tokens_details") or {}
    if isinstance(details.get("reasoning_tokens"), int):
        out["reasoning_tokens"] = details["reasoning_tokens"]
    return out


def _retry_after(response: httpx.Response) -> float | None:
    """Retry-After header, else the ``retryDelay`` hint Gemini puts in 429 bodies."""
    try:
        return max(0.0, float(response.headers.get("retry-after", "")))
    except ValueError:
        pass
    match = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', response.text)
    return float(match.group(1)) if match else None


def _reply(payload: dict[str, Any]) -> _Reply:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    return _Reply(
        text=content if isinstance(content, str) else str(content),
        reasoning=message_reasoning(message),
        finish_reason=choice.get("finish_reason"),
        payload=payload,
    )


class ChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        concurrency: int = 8,
        endpoint_kind: str = "hosted",
        stream: bool = False,
        deadline_seconds: float = 180,
        request_timeout: float = 120,
        max_attempts: int = 6,
        requests_per_minute: float | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if endpoint_kind not in MAX_CONCURRENCY:
            raise ValueError(f"endpoint_kind must be one of {', '.join(ENDPOINT_KINDS)}")
        limit = MAX_CONCURRENCY[endpoint_kind]
        if not 1 <= concurrency <= limit:
            raise ValueError(
                f"Evaluation LLM concurrency must be between 1 and {limit} for "
                f"{endpoint_kind} endpoints"
            )
        if not base_url.startswith(("https://", "http://127.0.0.1", "http://localhost")):
            raise ValueError("LLM endpoint must use HTTPS outside loopback")
        if requests_per_minute is not None and requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        if endpoint_kind == "self-hosted" and requests_per_minute is not None:
            raise ValueError("Self-hosted endpoints are not paced; omit requests_per_minute")
        if deadline_seconds <= 0 or request_timeout <= 0:
            raise ValueError("deadline_seconds and request_timeout must be positive")
        self.endpoint_kind = endpoint_kind
        self.stream = stream
        self._base = base_url.rstrip("/")
        self._url = self._base + "/chat/completions"
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout, connect=10),
            headers={"Authorization": f"Bearer {api_key}"},
            follow_redirects=False,
            limits=httpx.Limits(max_connections=concurrency + 4),
        )
        self._semaphore = asyncio.Semaphore(concurrency)
        self._deadline = deadline_seconds
        self._max_attempts = max_attempts
        self._rng = rng or random.Random(0)
        self._interval = 60 / requests_per_minute if requests_per_minute else 0.0
        self._next_slot: dict[str, float] = {}
        self._pace_lock = asyncio.Lock()
        self.stats = CallStats()

    async def probe(self, *, ready_timeout: float = 900, poll_seconds: float = 5) -> dict[str, Any]:
        """Wait until ``GET /models`` succeeds; return served model ids and server version.

        Scale-to-zero servers answer only after a cold start, so this polls through
        transport errors, 5xx and the 303 redirects some platforms send for slow
        requests. Non-retryable client errors (for example 401) fail immediately.
        """
        started = time.perf_counter()
        attempts = 0
        last = "no attempt"
        while True:
            attempts += 1
            try:
                response = await self._client.get(self._base + "/models")
                if response.status_code == 200:
                    data = response.json().get("data") or []
                    models = sorted(str(item.get("id")) for item in data if isinstance(item, dict))
                    return {
                        "models": models,
                        "version": await self._server_version(),
                        "ready_seconds": round(time.perf_counter() - started, 1),
                        "probe_attempts": attempts,
                    }
                last = f"http_{response.status_code}"
                if response.status_code in {400, 401, 403, 404}:
                    raise RuntimeError(f"LLM endpoint readiness probe failed with {last}")
            except (httpx.TransportError, ValueError) as exc:
                last = type(exc).__name__
            if time.perf_counter() - started + poll_seconds > ready_timeout:
                raise TimeoutError(f"LLM endpoint not ready after {ready_timeout:.0f}s ({last})")
            await asyncio.sleep(poll_seconds)

    async def _server_version(self) -> str | None:
        root = self._base.removesuffix("/v1")
        try:
            response = await self._client.get(root + "/version")
            if response.status_code == 200:
                version = response.json().get("version")
                return str(version) if version is not None else None
        except (httpx.TransportError, ValueError):
            pass
        return None

    async def _pace(self, model: str) -> None:
        """Per-model client-side pacing so provider quotas are not hit by design."""
        if not self._interval:
            return
        async with self._pace_lock:
            now = time.monotonic()
            slot = max(now, self._next_slot.get(model, now))
            self._next_slot[model] = slot + self._interval
        await asyncio.sleep(slot - now)

    def _sample_error(self, response: httpx.Response) -> None:
        text = " ".join(response.text.split())
        quota = re.findall(r'"(quotaMetric|quotaId|quotaValue|retryDelay)"\s*:\s*"([^"]*)"', text)
        detail = "; ".join(f"{k}={v}" for k, v in quota) or text[:300]
        sample = f"{response.status_code}: {detail}"
        if len(self.stats.error_samples) < 5 and sample not in self.stats.error_samples:
            self.stats.error_samples.append(sample)

    def _count(self, status: str) -> None:
        self.stats.status_counts[status] = self.stats.status_counts.get(status, 0) + 1

    async def _read_stream(self, response: httpx.Response, sent: float) -> _Reply:
        """Accumulate an SSE chat completion; usage arrives in the final chunk."""
        content: list[str] = []
        reasoning: list[str] = []
        finish_reason: str | None = None
        usage: dict[str, Any] | None = None
        first: float | None = None
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise ValueError("stream error event")
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                text = delta.get("content")
                thought = message_reasoning(delta)
                if (text or thought) and first is None:
                    first = time.perf_counter()
                if isinstance(text, str):
                    content.append(text)
                if thought:
                    reasoning.append(thought)
                finish_reason = choice.get("finish_reason") or finish_reason
        return _Reply(
            text="".join(content),
            reasoning="".join(reasoning),
            finish_reason=finish_reason,
            payload={"usage": usage} if usage else {},
            ttft_ms=round((first - sent) * 1000, 1) if first is not None else None,
        )

    async def complete(
        self, config: GenerationConfig, messages: list[dict[str, str]]
    ) -> Completion:
        body = config.body(messages)
        if self.stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        async with self._semaphore:
            await self._pace(config.model)
            start = time.perf_counter()
            deadline = start + self._deadline
            attempt = 0
            error = "unknown"
            while attempt < self._max_attempts:
                attempt += 1
                if attempt > 1:
                    await self._pace(config.model)
                self.stats.requests += 1
                delay: float | None = None
                sent = time.perf_counter()
                try:
                    async with asyncio.timeout(max(1.0, deadline - sent)):
                        async with self._client.stream("POST", self._url, json=body) as response:
                            self._count(str(response.status_code))
                            if response.status_code == 200:
                                reply = (
                                    await self._read_stream(response, sent)
                                    if self.stream
                                    else _reply(json.loads(await response.aread()))
                                )
                                now = time.perf_counter()
                                return Completion(
                                    text=reply.text,
                                    finish_reason=reply.finish_reason,
                                    usage=_usage(reply.payload),
                                    attempts=attempt,
                                    latency_ms=round((now - start) * 1000, 1),
                                    service_ms=round((now - sent) * 1000, 1),
                                    reasoning_chars=len(reply.reasoning),
                                    ttft_ms=reply.ttft_ms,
                                )
                            await response.aread()
                            error = f"http_{response.status_code}"
                            self._sample_error(response)
                            if response.status_code not in RETRYABLE_STATUS:
                                break
                            delay = _retry_after(response)
                except TimeoutError:
                    error = "deadline"
                    self._count(error)
                    break
                except (httpx.TransportError, ValueError) as exc:
                    error = type(exc).__name__
                    self._count(error)
                backoff = self._rng.uniform(0, min(30.0, 2.0**attempt))
                wait = max(delay or 0.0, backoff)
                if time.perf_counter() + wait >= deadline or attempt >= self._max_attempts:
                    break
                self.stats.retries += 1
                await asyncio.sleep(wait)
            self.stats.failures += 1
            return Completion(
                text="",
                finish_reason=None,
                usage={},
                attempts=attempt,
                latency_ms=round((time.perf_counter() - start) * 1000, 1),
                error=error,
            )

    async def close(self) -> None:
        await self._client.aclose()
