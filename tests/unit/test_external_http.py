import asyncio
import gzip
from decimal import Decimal

import httpx
import pytest
import structlog

from ledger_rag.core.retry import RetryPolicy
from ledger_rag.external.breaker import CircuitBreaker
from ledger_rag.external.errors import ExternalDataError
from ledger_rag.external.http import (
    ExternalHttp,
    ResponseCache,
    SlidingWindowLimiter,
    parse_json,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.slept.append(delay)
        self.now += delay


def make_http(
    handler: httpx.MockTransport,
    *,
    clock: FakeClock | None = None,
    cache: ResponseCache | None = None,
    max_bytes: int = 1_000,
    retries: int = 2,
) -> ExternalHttp:
    clock = clock or FakeClock()
    return ExternalHttp(
        provider="SEC EDGAR",
        client=httpx.AsyncClient(transport=handler),
        limiter=SlidingWindowLimiter(100, 1.0, clock=clock, sleep=clock.sleep),
        cache=cache or ResponseCache(60, 10_000, clock=clock),
        policy=RetryPolicy(max_retries=retries, base_delay=0.1, max_delay=1.0),
        max_bytes=max_bytes,
    )


async def test_limiter_spaces_request_starts_over_the_window() -> None:
    clock = FakeClock()
    limiter = SlidingWindowLimiter(2, 1.0, clock=clock, sleep=clock.sleep)
    for _ in range(5):
        await limiter.acquire()
    assert clock.slept == [1.0, 1.0]
    assert clock.now == 2.0


async def test_limiter_serves_waiters_in_arrival_order() -> None:
    clock = FakeClock()
    limiter = SlidingWindowLimiter(1, 1.0, clock=clock, sleep=clock.sleep)
    order: list[int] = []

    async def take(index: int) -> None:
        await limiter.acquire()
        order.append(index)

    await asyncio.gather(*(take(index) for index in range(4)))
    assert order == [0, 1, 2, 3]
    assert clock.now == 3.0


def test_limiter_rejects_nonsense_limits() -> None:
    with pytest.raises(ValueError):
        SlidingWindowLimiter(0, 1.0)


def test_cache_is_byte_bounded_lru_with_ttl() -> None:
    clock = FakeClock()
    cache = ResponseCache(10, 100, clock=clock)
    cache.put("a", "A", 40)
    cache.put("b", "B", 40)
    assert cache.get("a") is not None  # touch: b becomes least recently used
    cache.put("c", "C", 40)
    assert cache.get("b") is None
    assert cache.size == 80
    cache.put("huge", "H", 101)
    assert cache.get("huge") is None
    entry = cache.get("a")
    assert entry is not None and cache.fresh(entry)
    clock.now = 11
    assert not cache.fresh(entry)
    cache.refresh("a", entry)
    assert cache.fresh(entry)
    cache.put("short", "S", 1, ttl=3600)
    short = cache.get("short")
    assert short is not None and short.expires == clock.now + 10  # never above the TTL


def test_cache_disabled_with_zero_ttl() -> None:
    cache = ResponseCache(0, 100)
    cache.put("a", "A", 1)
    assert cache.get("a") is None


def test_json_numbers_stay_decimal() -> None:
    assert parse_json(b'{"val": 0.1, "n": 3}') == {"val": Decimal("0.1"), "n": 3}


async def test_fresh_cache_hit_skips_the_network() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=b'{"ok": 1}')

    http = make_http(httpx.MockTransport(handler))
    for _ in range(3):
        assert await http.get("https://data.sec.gov/x", key="k", endpoint="x", parse=parse_json)
    assert len(calls) == 1


async def test_conditional_revalidation_reuses_the_cached_value() -> None:
    clock = FakeClock()
    seen: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.headers))
        if "if-none-match" in request.headers:
            return httpx.Response(304)
        return httpx.Response(
            200,
            content=b'{"v": 1}',
            headers={"etag": '"abc"', "last-modified": "Wed, 01 Jul 2026 00:00:00 GMT"},
        )

    http = make_http(httpx.MockTransport(handler), clock=clock)
    first = await http.get("https://www.sec.gov/x", key="k", endpoint="x", parse=parse_json)
    clock.now = 120
    second = await http.get("https://www.sec.gov/x", key="k", endpoint="x", parse=parse_json)
    assert first == second == {"v": 1}
    assert seen[1]["if-none-match"] == '"abc"'
    assert seen[1]["if-modified-since"] == "Wed, 01 Jul 2026 00:00:00 GMT"
    clock.now = 150  # the 304 renewed the entry
    await http.get("https://www.sec.gov/x", key="k", endpoint="x", parse=parse_json)
    assert len(seen) == 2


async def test_concurrent_callers_share_one_request() -> None:
    release = asyncio.Event()
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await release.wait()
        return httpx.Response(200, content=b"[1]")

    http = make_http(httpx.MockTransport(handler))
    tasks = [
        asyncio.create_task(
            http.get("https://data.sec.gov/x", key="k", endpoint="x", parse=parse_json)
        )
        for _ in range(5)
    ]
    await asyncio.sleep(0)
    release.set()
    assert await asyncio.gather(*tasks) == [[1]] * 5
    assert calls == 1


async def test_cancelled_caller_does_not_abort_the_shared_fetch() -> None:
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        await release.wait()
        return httpx.Response(200, content=b"[2]")

    http = make_http(httpx.MockTransport(handler))
    first = asyncio.create_task(
        http.get("https://data.sec.gov/x", key="k", endpoint="x", parse=parse_json)
    )
    second = asyncio.create_task(
        http.get("https://data.sec.gov/x", key="k", endpoint="x", parse=parse_json)
    )
    await asyncio.sleep(0.01)
    first.cancel()
    release.set()
    assert await second == [2]
    with pytest.raises(asyncio.CancelledError):
        await first


async def test_retries_429_and_5xx_with_backoff(no_retry_sleep: list[float]) -> None:
    statuses = iter([429, 503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return httpx.Response(status, content=b'{"ok": true}' if status == 200 else b"busy")

    http = make_http(httpx.MockTransport(handler))
    assert await http.get("https://data.sec.gov/x", key="k", endpoint="x", parse=parse_json)
    assert len(no_retry_sleep) == 2


@pytest.mark.parametrize(
    ("status", "kind", "message"),
    [
        (404, "not_found", "SEC EDGAR has no data for that request"),
        (429, "rate_limited", "SEC EDGAR is rate limiting requests"),
        (403, "upstream", "SEC EDGAR refused the request"),
        (500, "upstream", "SEC EDGAR returned HTTP 500"),
    ],
)
async def test_status_errors_are_sanitized(status: int, kind: str, message: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=b"<html>secret internals</html>")

    http = make_http(httpx.MockTransport(handler), retries=0)
    with pytest.raises(ExternalDataError) as caught:
        await http.get(
            "https://data.sec.gov/x?api_key=hidden", key="k", endpoint="x", parse=parse_json
        )
    assert (caught.value.kind, caught.value.message) == (kind, message)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


async def test_not_found_is_negative_cached() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404)

    http = make_http(httpx.MockTransport(handler))
    for _ in range(2):
        with pytest.raises(ExternalDataError, match="no data"):
            await http.get("https://data.sec.gov/x", key="k", endpoint="x", parse=parse_json)
    assert calls == 1


async def test_declared_and_streamed_size_caps() -> None:
    def declared(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"[]", headers={"content-length": "5000"})

    def streamed(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"[" + b"1," * 2000 + b"1]")

    def compressed(request: httpx.Request) -> httpx.Response:
        body = gzip.compress(b"[" + b"0," * 5000 + b"0]")
        return httpx.Response(200, content=body, headers={"content-encoding": "gzip"})

    for handler in (declared, streamed, compressed):
        http = make_http(httpx.MockTransport(handler))
        with pytest.raises(ExternalDataError) as caught:
            await http.get("https://data.sec.gov/x", key="k", endpoint="x", parse=parse_json)
        assert caught.value.kind == "too_large"


async def test_malformed_body_is_an_upstream_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{not json")

    http = make_http(httpx.MockTransport(handler))
    with pytest.raises(ExternalDataError) as caught:
        await http.get("https://data.sec.gov/x", key="k", endpoint="x", parse=parse_json)
    assert (caught.value.kind, caught.value.message) == (
        "upstream",
        "SEC EDGAR returned an unexpected response",
    )


async def test_transport_errors_never_leak_urls() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out at https://api.stlouisfed.org/?api_key=hidden")

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused https://api.stlouisfed.org/?api_key=hidden")

    for handler, kind in ((timeout, "timeout"), (refused, "upstream")):
        http = make_http(httpx.MockTransport(handler), retries=0)
        with pytest.raises(ExternalDataError) as caught:
            await http.get("https://data.sec.gov/x", key="k", endpoint="x", parse=parse_json)
        assert caught.value.kind == kind
        assert "hidden" not in str(caught.value) and caught.value.__cause__ is None


async def test_request_logs_carry_no_url_or_query() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{}")

    http = make_http(httpx.MockTransport(handler))
    with structlog.testing.capture_logs() as logs:
        await http.get(
            "https://api.stlouisfed.org/fred/series",
            key="fred:series",
            endpoint="series",
            params={"api_key": "hidden"},
            parse=parse_json,
        )
    assert logs and "hidden" not in repr(logs)
    assert logs[-1]["event"] == "external_request" and logs[-1]["status"] == 200


async def test_provider_error_hook_maps_bodies() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, content=b'{"error_message": "nope"}')

    http = make_http(httpx.MockTransport(handler))
    with pytest.raises(ExternalDataError) as caught:
        await http.get(
            "https://x.test",
            key="k",
            endpoint="x",
            parse=parse_json,
            on_error=lambda status, body: ExternalDataError("invalid_request", f"bad {status}"),
        )
    assert caught.value.message == "bad 400"


def test_breaker_opens_half_opens_and_closes() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker("SEC EDGAR", 2, 30, clock=clock)
    breaker.failure(ExternalDataError("timeout", "slow"))
    assert breaker.state == "closed"
    breaker.failure(ExternalDataError("upstream", "500"))
    assert breaker.state == "open"
    with pytest.raises(ExternalDataError) as caught:
        breaker.before()
    assert caught.value.kind == "circuit_open"
    clock.now = 30
    assert breaker.state == "half_open"
    breaker.before()  # the single probe
    with pytest.raises(ExternalDataError):
        breaker.before()
    breaker.failure(ExternalDataError("rate_limited", "429"))
    assert breaker.state == "open"
    clock.now = 60
    breaker.before()
    breaker.success()
    assert breaker.state == "closed"


def test_breaker_ignores_caller_errors_and_cancellations() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker("FRED", 1, 30, clock=clock)
    breaker.failure(ExternalDataError("not_found", "unknown series"))
    breaker.failure(ExternalDataError("invalid_request", "bad period"))
    assert breaker.state == "closed"
    breaker.failure(ExternalDataError("too_large", "big"))
    clock.now = 31
    breaker.before()
    breaker.release()
    breaker.before()  # a cancelled probe frees the slot for the next caller
    breaker.failure(ExternalDataError("not_found", "unknown series"))
    assert breaker.state == "closed"
