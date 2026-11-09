"""Deadline-aware retries with full jitter for idempotent upstream HTTP calls.

Only the provider, embedding and reranker requests use this. They are pure
computations, so repeating them is safe; a retried LLM call can still be billed
twice. Qdrant writes are never retried here: their failure semantics are owned by
the write lease in ``retrieval.qdrant``.

Request handlers open a :func:`retry_budget` scope holding the request deadline and
an optional retry cap (voice allows at most one retry). A retry happens only when
the backoff delay plus a minimum useful attempt still fits before that deadline.
Calls outside any scope (scripts, ingestion) use the configured policy without a
deadline.
"""

import asyncio
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from email.utils import parsedate_to_datetime

import httpx
import structlog

from ledger_rag.core.config import Settings

logger = structlog.get_logger(__name__)

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 529})
MIN_ATTEMPT_SECONDS = 0.25
_sleep = asyncio.sleep


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2
    base_delay: float = 0.2
    max_delay: float = 5.0

    @classmethod
    def from_settings(cls, settings: Settings) -> "RetryPolicy":
        return cls(
            settings.upstream_max_retries,
            settings.upstream_retry_base_seconds,
            settings.upstream_retry_max_delay_seconds,
        )


@dataclass(frozen=True)
class _Budget:
    deadline: float
    max_retries: int | None


_budget: ContextVar[_Budget | None] = ContextVar("ledger_retry_budget", default=None)


@contextmanager
def retry_budget(seconds: float, max_retries: int | None = None) -> Iterator[None]:
    """Bound retries in this context to ``seconds`` from now and ``max_retries``."""
    token = _budget.set(_Budget(time.monotonic() + seconds, max_retries))
    try:
        yield
    finally:
        _budget.reset(token)


VOICE_MAX_RETRIES = 1


@asynccontextmanager
async def request_deadline(seconds: float, *, voice: bool = False) -> AsyncIterator[None]:
    """Enforce a request deadline and scope upstream retries to it (voice: one retry)."""
    async with asyncio.timeout(seconds):
        with retry_budget(seconds, VOICE_MAX_RETRIES if voice else None):
            yield


def retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after", "").strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


async def send_with_retries(
    send: Callable[[], Awaitable[httpx.Response]], policy: RetryPolicy, *, operation: str
) -> httpx.Response:
    """Send, retrying 408/425/429/5xx-overload and transport errors within budget.

    Returns the last response (callers keep their own status handling) or re-raises
    the last transport error.
    """
    budget = _budget.get()
    retries = policy.max_retries
    if budget is not None and budget.max_retries is not None:
        retries = min(retries, budget.max_retries)
    attempt = 0
    while True:
        status: int | None = None
        try:
            response = await send()
        except httpx.TransportError:
            if attempt >= retries:
                raise
            delay = _backoff(policy, attempt)
            if not _fits(budget, delay):
                raise
        else:
            if response.status_code not in RETRYABLE_STATUS or attempt >= retries:
                return response
            status = response.status_code
            delay = _backoff(policy, attempt)
            requested = retry_after_seconds(response)
            if requested is not None:
                if requested > policy.max_delay:
                    return response
                delay = max(delay, requested)
            if not _fits(budget, delay):
                return response
            await response.aclose()
        attempt += 1
        logger.info(
            "upstream_retry",
            operation=operation,
            attempt=attempt,
            status=status,
            delay_ms=round(delay * 1000),
        )
        await _sleep(delay)


def _backoff(policy: RetryPolicy, attempt: int) -> float:
    return random.uniform(0, min(policy.max_delay, policy.base_delay * 2**attempt))  # noqa: S311


def _fits(budget: _Budget | None, delay: float) -> bool:
    if budget is None:
        return True
    return time.monotonic() + delay + MIN_ATTEMPT_SECONDS <= budget.deadline
