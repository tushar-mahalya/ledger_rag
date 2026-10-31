"""Bounded, deadline-aware OpenAI-compatible chat client for offline evaluation.

The production providers deliberately expose a narrow request surface. Benchmarks
need explicit, recorded generation settings (temperature, token cap, seed,
reasoning effort), token usage, and retry accounting, so evaluation uses this
small client instead. It never logs request bodies, keys or response bodies.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class GenerationConfig:
    model: str
    temperature: float | None = 0.0
    max_tokens: int = 2048
    reasoning_effort: str | None = None
    seed: int | None = None

    @property
    def label(self) -> str:
        return f"{self.model}@{self.reasoning_effort}" if self.reasoning_effort else self.model


@dataclass
class Completion:
    text: str
    finish_reason: str | None
    usage: dict[str, int]
    attempts: int
    latency_ms: float
    error: str | None = None


@dataclass
class CallStats:
    requests: int = 0
    retries: int = 0
    failures: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)


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
    try:
        return max(0.0, float(response.headers.get("retry-after", "")))
    except ValueError:
        return None


class ChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        concurrency: int = 8,
        deadline_seconds: float = 180,
        request_timeout: float = 120,
        max_attempts: int = 6,
        rng: random.Random | None = None,
    ) -> None:
        if not 1 <= concurrency <= 8:
            raise ValueError("Evaluation LLM concurrency must be between 1 and 8")
        if not base_url.startswith(("https://", "http://127.0.0.1", "http://localhost")):
            raise ValueError("LLM endpoint must use HTTPS outside loopback")
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout, connect=10),
            headers={"Authorization": f"Bearer {api_key}"},
            follow_redirects=False,
        )
        self._semaphore = asyncio.Semaphore(concurrency)
        self._deadline = deadline_seconds
        self._max_attempts = max_attempts
        self._rng = rng or random.Random(0)
        self.stats = CallStats()

    def _count(self, status: str) -> None:
        self.stats.status_counts[status] = self.stats.status_counts.get(status, 0) + 1

    async def complete(
        self, config: GenerationConfig, messages: list[dict[str, str]]
    ) -> Completion:
        body: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "max_tokens": config.max_tokens,
        }
        if config.temperature is not None:
            body["temperature"] = config.temperature
        if config.reasoning_effort:
            body["reasoning_effort"] = config.reasoning_effort
        if config.seed is not None:
            body["seed"] = config.seed
        async with self._semaphore:
            start = time.perf_counter()
            deadline = start + self._deadline
            attempt = 0
            error = "unknown"
            while attempt < self._max_attempts:
                attempt += 1
                self.stats.requests += 1
                delay: float | None = None
                try:
                    response = await self._client.post(self._url, json=body)
                    self._count(str(response.status_code))
                    if response.status_code == 200:
                        payload = response.json()
                        choice = (payload.get("choices") or [{}])[0]
                        message = choice.get("message") or {}
                        content = message.get("content") or ""
                        return Completion(
                            text=content if isinstance(content, str) else str(content),
                            finish_reason=choice.get("finish_reason"),
                            usage=_usage(payload),
                            attempts=attempt,
                            latency_ms=round((time.perf_counter() - start) * 1000, 1),
                        )
                    error = f"http_{response.status_code}"
                    if response.status_code not in RETRYABLE_STATUS:
                        break
                    delay = _retry_after(response)
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
