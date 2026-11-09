"""Polite, bounded HTTP for official public data APIs.

Every request goes through a per-provider sliding-window limiter (SEC: at most 10
requests/second; FRED: at most 120/minute), a TTL + byte-bounded LRU cache with
conditional revalidation and single-flight de-duplication, deadline-aware retries with
full jitter, a decoded-size cap and JSON parsing that keeps every number a ``Decimal``.

Failures surface as :class:`ExternalDataError` with fixed messages: upstream URLs (FRED
carries its key in the query string), bodies and exception chains never leave here.
Limits are per process; divide the configured rate by the replica count.
"""

import asyncio
import json
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Generic, TypeVar

import httpx
import structlog

from ledger_rag.core.retry import RetryPolicy, request_deadline, send_with_retries
from ledger_rag.external.errors import ExternalDataError

logger = structlog.get_logger(__name__)

T = TypeVar("T")
Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]
NOT_FOUND_TTL_SECONDS = 300.0
# companyfacts documents run to several megabytes; parse those off the event loop.
THREAD_PARSE_BYTES = 1_000_000


class SlidingWindowLimiter:
    """At most ``limit`` request starts in any ``period``-second window (FIFO waiters)."""

    def __init__(
        self,
        limit: int,
        period: float,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if limit < 1 or period <= 0:
            raise ValueError("Limiter needs a positive limit and period")
        self.limit = limit
        self.period = period
        self._clock = clock
        self._sleep = sleep
        self._starts: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = self._clock()
                while self._starts and self._starts[0] <= now - self.period:
                    self._starts.popleft()
                if len(self._starts) < self.limit:
                    self._starts.append(now)
                    return
                await self._sleep(self._starts[0] + self.period - now)


@dataclass
class _Entry(Generic[T]):
    value: T | None
    size: int
    expires: float
    etag: str | None
    last_modified: str | None


class ResponseCache:
    """LRU keyed by logical request (never by secret-bearing URL), bounded by bytes."""

    def __init__(self, ttl: float, max_bytes: int, *, clock: Clock = time.monotonic) -> None:
        self.ttl = ttl
        self.max_bytes = max_bytes
        self._clock = clock
        self._entries: OrderedDict[str, _Entry[Any]] = OrderedDict()
        self._bytes = 0

    def get(self, key: str) -> _Entry[Any] | None:
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
        return entry

    def fresh(self, entry: _Entry[Any]) -> bool:
        return entry.expires > self._clock()

    def put(
        self,
        key: str,
        value: Any,
        size: int,
        *,
        ttl: float | None = None,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> None:
        self.drop(key)
        lifetime = self.ttl if ttl is None else min(ttl, self.ttl)
        if lifetime <= 0 or size > self.max_bytes:
            return
        self._entries[key] = _Entry(value, size, self._clock() + lifetime, etag, last_modified)
        self._bytes += size
        while self._bytes > self.max_bytes and self._entries:
            _, evicted = self._entries.popitem(last=False)
            self._bytes -= evicted.size

    def refresh(self, key: str, entry: _Entry[Any]) -> None:
        entry.expires = self._clock() + self.ttl

    def drop(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._bytes -= entry.size

    @property
    def size(self) -> int:
        return self._bytes


def parse_json(raw: bytes) -> Any:
    """Parse JSON with exact decimals; floats never enter financial values."""
    return json.loads(raw, parse_float=Decimal)


ErrorHook = Callable[[int, bytes], ExternalDataError | None]


@dataclass
class _Flight:
    task: asyncio.Task[Any]
    waiters: int = 0


async def _drain(tasks: list[asyncio.Task[Any]]) -> None:
    finished = asyncio.gather(*tasks, return_exceptions=True)
    interrupted = False
    while not finished.done():
        try:
            # A second caller cancellation must not interrupt stream cleanup.
            await asyncio.shield(finished)
        except asyncio.CancelledError:
            interrupted = True
    if interrupted:
        raise asyncio.CancelledError


class ExternalHttp:
    """One provider's transport: limiter, cache, retries, caps and sanitized errors."""

    def __init__(
        self,
        *,
        provider: str,
        client: httpx.AsyncClient,
        limiter: SlidingWindowLimiter,
        cache: ResponseCache,
        policy: RetryPolicy,
        max_bytes: int,
        timeout_seconds: float = 8.0,
    ) -> None:
        self.provider = provider
        self.client = client
        self.limiter = limiter
        self.cache = cache
        self.policy = policy
        self.max_bytes = max_bytes
        self.timeout_seconds = timeout_seconds
        self._inflight: dict[str, _Flight] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closed = False

    async def get(
        self,
        url: str,
        *,
        key: str,
        endpoint: str,
        params: Mapping[str, str] | None = None,
        parse: Callable[[bytes], T],
        on_error: ErrorHook | None = None,
    ) -> T:
        """Fetch and parse ``url``; ``key`` must identify the request without secrets."""
        if self._closed:
            raise ExternalDataError("disabled", f"{self.provider} transport is closed")
        entry = self.cache.get(key)
        if entry is not None and self.cache.fresh(entry):
            cached: T = self._cached(entry, endpoint)
            return cached
        flight = self._inflight.get(key)
        if flight is None:
            task = asyncio.create_task(
                self._bounded_fetch(url, key, endpoint, params, parse, on_error, entry),
                name=f"external.fetch:{self.provider}:{endpoint}",
            )
            flight = _Flight(task)
            self._inflight[key] = flight
            self._tasks.add(task)
            task.add_done_callback(lambda done: self._finish(key, done))
        flight.waiters += 1
        try:
            # A cancelled waiter must not cancel a request another waiter still needs.
            result: T = await asyncio.shield(flight.task)
            return result
        finally:
            flight.waiters -= 1
            if flight.waiters == 0 and not flight.task.done():
                if self._inflight.get(key) is flight:
                    del self._inflight[key]
                if not flight.task.cancelling():
                    flight.task.cancel()
                await _drain([flight.task])

    def _finish(self, key: str, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        flight = self._inflight.get(key)
        if flight is not None and flight.task is task:
            del self._inflight[key]
        if not task.cancelled():
            task.exception()

    async def aclose(self) -> None:
        self._closed = True
        tasks = list(self._tasks)
        self._inflight.clear()
        for task in tasks:
            if not task.cancelling():
                task.cancel()
        await _drain(tasks)

    async def _bounded_fetch(
        self,
        url: str,
        key: str,
        endpoint: str,
        params: Mapping[str, str] | None,
        parse: Callable[[bytes], T],
        on_error: ErrorHook | None,
        entry: _Entry[Any] | None,
    ) -> T:
        try:
            # A fresh budget belongs to the shared fetch, not to its first waiter.
            async with request_deadline(self.timeout_seconds):
                return await self._fetch(url, key, endpoint, params, parse, on_error, entry)
        except TimeoutError:
            raise ExternalDataError("timeout", f"{self.provider} did not respond in time") from None

    def _cached(self, entry: _Entry[Any], endpoint: str) -> Any:
        if entry.value is None:
            raise ExternalDataError("not_found", f"{self.provider} has no data for that request")
        logger.debug("external_cache_hit", provider=self.provider, endpoint=endpoint)
        return entry.value

    async def _fetch(
        self,
        url: str,
        key: str,
        endpoint: str,
        params: Mapping[str, str] | None,
        parse: Callable[[bytes], T],
        on_error: ErrorHook | None,
        stale: _Entry[Any] | None,
    ) -> T:
        headers: dict[str, str] = {}
        if stale is not None and stale.etag:
            headers["If-None-Match"] = stale.etag
        if stale is not None and stale.last_modified:
            headers["If-Modified-Since"] = stale.last_modified
        request = self.client.build_request("GET", url, params=params, headers=headers)

        async def send() -> httpx.Response:
            await self.limiter.acquire()
            return await self.client.send(request, stream=True)

        started = time.perf_counter()
        status: int | None = None
        try:
            response = await send_with_retries(
                send, self.policy, operation=f"{self.provider.lower()}_{endpoint}"
            )
            status = response.status_code
            try:
                if status == 304 and stale is not None:
                    self.cache.refresh(key, stale)
                    revalidated: T = self._cached(stale, endpoint)
                    return revalidated
                raw = await self._read(response)
            finally:
                await response.aclose()
        except httpx.TimeoutException:
            raise ExternalDataError("timeout", f"{self.provider} did not respond in time") from None
        except httpx.HTTPError:
            raise ExternalDataError("upstream", f"{self.provider} is unreachable") from None
        finally:
            logger.info(
                "external_request",
                provider=self.provider,
                endpoint=endpoint,
                status=status,
                duration_ms=round((time.perf_counter() - started) * 1000),
            )
        if status == 200:
            try:
                value = (
                    await asyncio.to_thread(parse, raw)
                    if len(raw) > THREAD_PARSE_BYTES
                    else parse(raw)
                )
            except (ValueError, TypeError, KeyError) as exc:
                logger.warning(
                    "external_malformed_response",
                    provider=self.provider,
                    endpoint=endpoint,
                    error=type(exc).__name__,
                )
                raise ExternalDataError(
                    "upstream", f"{self.provider} returned an unexpected response"
                ) from None
            self.cache.put(
                key,
                value,
                len(raw),
                etag=response.headers.get("etag"),
                last_modified=response.headers.get("last-modified"),
            )
            return value
        error = on_error(status, raw) if on_error is not None else None
        if error is None:
            error = self._status_error(status)
        if error.kind == "not_found":
            self.cache.put(key, None, 64, ttl=NOT_FOUND_TTL_SECONDS)
        raise error

    async def _read(self, response: httpx.Response) -> bytes:
        declared = response.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > self.max_bytes:
            raise ExternalDataError("too_large", f"{self.provider} response exceeds the size cap")
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > self.max_bytes:
                raise ExternalDataError(
                    "too_large", f"{self.provider} response exceeds the size cap"
                )
        return bytes(body)

    def _status_error(self, status: int) -> ExternalDataError:
        if status == 404:
            return ExternalDataError("not_found", f"{self.provider} has no data for that request")
        if status == 429:
            return ExternalDataError("rate_limited", f"{self.provider} is rate limiting requests")
        if status == 403:
            return ExternalDataError("upstream", f"{self.provider} refused the request")
        return ExternalDataError("upstream", f"{self.provider} returned HTTP {status}")
