import asyncio
from datetime import date

import httpx
import pytest

from ledger_rag.core.retry import request_deadline
from ledger_rag.external.errors import ExternalDataError
from ledger_rag.external.fred import SeriesMeta
from ledger_rag.external.service import ExternalSession
from ledger_rag.external.tools import run_external_tool
from tests.unit.external_fakes import FakeUpstream, external_data, fixture
from tests.unit.test_external_http import make_http, parse_json
from tests.unit.test_external_service import READER


@pytest.mark.parametrize(
    ("series", "wanted", "cutoff"),
    [
        ("FEDFUNDS", "2023-03-01", "2023-03-15"),
        ("GDP", "2023-07-01", "2023-08-01"),
    ],
)
async def test_unpublished_period_is_not_replaced_by_previous_period(
    series: str, wanted: str, cutoff: str
) -> None:
    data = external_data()
    try:
        result = await data.macro_series(
            series, as_of=date.fromisoformat(cutoff), dates=[date.fromisoformat(wanted)]
        )
        assert len(result.facts) == 0
        assert [miss.period for miss in result.missing] == [wanted]
    finally:
        await data.aclose()


class DripStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = asyncio.Event()
        self.started = asyncio.Event()

    async def __aiter__(self):
        self.started.set()
        while True:
            await asyncio.sleep(0.005)
            yield b" "

    async def aclose(self) -> None:
        self.closed.set()


async def cleanup_fetches(http) -> None:
    # Clean up even against the broken implementation so repro runs do not leak tasks.
    tasks = [getattr(item, "task", item) for item in list(http._inflight.values())]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await http.client.aclose()


async def test_lookup_timeout_cancels_orphan_stream_before_return() -> None:
    stream = DripStream()
    upstream = FakeUpstream()
    upstream.overrides["company_tickers"] = lambda request: httpx.Response(200, stream=stream)
    data = external_data(upstream, external_timeout_seconds=0.02, fred_enabled=False)
    assert data.sec is not None
    try:
        with pytest.raises(ExternalDataError) as caught:
            await data.company_facts("AAPL", ["Assets"], ["FY2023"], as_of=date(2024, 1, 1))
        assert caught.value.kind == "timeout"
        assert stream.started.is_set()
        assert stream.closed.is_set()
        assert not data.sec.http._inflight
        assert not data.sec.http._tasks
    finally:
        await cleanup_fetches(data.sec.http)
        await data.aclose()


async def test_shared_fetch_has_independent_total_deadline() -> None:
    cancelled = asyncio.Event()

    async def blocked(request: httpx.Request) -> httpx.Response:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return httpx.Response(200, content=b"[]")

    http = make_http(httpx.MockTransport(blocked))
    http.timeout_seconds = 0.02
    try:
        with pytest.raises(ExternalDataError) as caught:
            async with asyncio.timeout(0.1):
                await http.get("https://data.sec.gov/x", key="k", endpoint="x", parse=parse_json)
        assert caught.value.kind == "timeout"
        assert cancelled.is_set()
        assert not http._inflight
        assert not http._tasks
    finally:
        await cleanup_fetches(http)


async def test_shutdown_cancels_and_awaits_fetch_with_waiter() -> None:
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def blocked(request: httpx.Request) -> httpx.Response:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return httpx.Response(200, content=b"{}")

    upstream = FakeUpstream()
    upstream.overrides["company_tickers"] = blocked
    data = external_data(upstream, fred_enabled=False)
    assert data.sec is not None
    call = asyncio.create_task(
        data.company_facts("AAPL", ["Assets"], ["FY2023"], as_of=date(2024, 1, 1))
    )
    try:
        await asyncio.wait_for(entered.wait(), 0.2)
        await data.aclose()
        assert cancelled.is_set()
        assert not data.sec.http._inflight
        assert not data.sec.http._tasks
        assert all(client.is_closed for client in data._clients)
        with pytest.raises(asyncio.CancelledError):
            await call
    finally:
        call.cancel()
        await asyncio.gather(call, return_exceptions=True)
        await cleanup_fetches(data.sec.http)
        await data.aclose()


@pytest.mark.parametrize("concept", ["Assets", "InterestExpense"])
async def test_cumulative_sec_tools_keep_distinct_currency_identity_and_readback(
    concept: str,
) -> None:
    upstream = FakeUpstream()
    payload = fixture("sec", f"concept_0000320193_us-gaap_{concept}.json")
    payload["units"]["EUR"] = [{**row, "val": row["val"] / 2} for row in payload["units"]["USD"]]
    upstream.overrides[concept] = lambda request: httpx.Response(200, json=payload)
    data = external_data(upstream, fred_enabled=False)
    session = ExternalSession(data, READER, date(2024, 1, 1), data.providers(READER))
    citations, ids = [], set()
    try:
        for unit in ["USD", "EUR"]:
            result = await run_external_tool(
                "lookup_company_facts",
                {
                    "company": "AAPL",
                    "concepts": [concept],
                    "periods": ["FY2023"],
                    "unit": unit,
                },
                session,
                citations,
                ids,
            )
            assert unit in result["evidence"][0]["text"]
        assert len(citations) == len(ids) == 2
        assert [item.id for item in citations] == ["C1", "C2"]
        assert citations[0].document_id != citations[1].document_id
        for citation, unit in zip(citations, ["USD", "EUR"], strict=True):
            assert citation.provenance is not None and citation.provenance.unit == unit
            fact = await data.read(citation.document_id, as_of=session.as_of)
            assert fact.quote == citation.quote and fact.provenance.unit == unit
        # Unitless legacy IDs must fail closed when the pinned filing has several units.
        with pytest.raises(ExternalDataError) as missing:
            await data.read(citations[0].document_id.rsplit(":", 1)[0], as_of=session.as_of)
        assert missing.value.kind == "not_found"
    finally:
        await data.aclose()


@pytest.mark.parametrize(
    ("frequency", "end"),
    [("M", "2023-01-31"), ("Q", "2023-03-31"), ("SA", "2023-06-30"), ("A", "2023-12-31")],
)
def test_macro_period_coverage_uses_calendar_bounds(frequency: str, end: str) -> None:
    from datetime import timedelta

    meta = SeriesMeta("TEST", "Test", frequency, frequency, "Units", "U", "NSA")
    start, last = date(2023, 1, 1), date.fromisoformat(end)
    assert meta.covers(start, last)
    assert not meta.covers(start, last + timedelta(days=1))


def test_daily_holiday_fallback_is_separate_and_bounded() -> None:
    meta = SeriesMeta("TEST", "Test", "Daily", "D", "Units", "U", "NSA")
    assert meta.covers(date(2023, 9, 15), date(2023, 9, 17))
    assert not meta.covers(date(2023, 9, 15), date(2023, 9, 22))
    assert not meta.covers(date(2023, 9, 18), date(2023, 9, 17))


async def test_one_surviving_waiter_keeps_fetch_after_first_deadline() -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return httpx.Response(200, content=b"[]")

    http = make_http(httpx.MockTransport(handler))
    http.timeout_seconds = 0.2

    async def caller(seconds: float):
        async with request_deadline(seconds):
            return await http.get(
                "https://data.sec.gov/x", key="shared", endpoint="x", parse=parse_json
            )

    first = asyncio.create_task(caller(0.02))
    second = asyncio.create_task(caller(0.15))
    try:
        await entered.wait()
        with pytest.raises(TimeoutError):
            await first
        assert len(http._inflight) == 1
        release.set()
        assert await second == []
        assert calls == 1
        assert not http._inflight and not http._tasks
    finally:
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
        await cleanup_fetches(http)


async def test_last_of_two_waiters_cancels_and_drains_stream() -> None:
    stream = DripStream()
    http = make_http(httpx.MockTransport(lambda request: httpx.Response(200, stream=stream)))
    first = asyncio.create_task(
        http.get("https://data.sec.gov/x", key="shared", endpoint="x", parse=parse_json)
    )
    second = asyncio.create_task(
        http.get("https://data.sec.gov/x", key="shared", endpoint="x", parse=parse_json)
    )
    try:
        await stream.started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not stream.closed.is_set()
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert stream.closed.is_set()
        assert not http._inflight and not http._tasks
        await http.aclose()
        with pytest.raises(ExternalDataError) as closed:
            await http.get("https://data.sec.gov/x", key="other", endpoint="x", parse=parse_json)
        assert closed.value.kind == "disabled"
    finally:
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
        await cleanup_fetches(http)


async def test_shutdown_awaits_orphan_already_draining_outside_coalescing_map() -> None:
    closing, release = asyncio.Event(), asyncio.Event()

    class SlowCloseStream(DripStream):
        async def aclose(self) -> None:
            closing.set()
            await release.wait()
            self.closed.set()

    stream = SlowCloseStream()
    upstream = FakeUpstream()
    upstream.overrides["company_tickers"] = lambda request: httpx.Response(200, stream=stream)
    data = external_data(upstream, external_timeout_seconds=0.02, fred_enabled=False)
    assert data.sec is not None
    call = asyncio.create_task(
        data.company_facts("AAPL", ["Assets"], ["FY2023"], as_of=date(2024, 1, 1))
    )
    shutdown = None
    try:
        await asyncio.wait_for(closing.wait(), 0.2)
        assert not data.sec.http._inflight and data.sec.http._tasks
        shutdown = asyncio.create_task(data.aclose())
        await asyncio.sleep(0.01)
        assert not shutdown.done()
        release.set()
        with pytest.raises(ExternalDataError) as error:
            await call
        assert error.value.kind == "timeout"
        await shutdown
        assert stream.closed.is_set() and not data.sec.http._tasks
    finally:
        release.set()
        call.cancel()
        await asyncio.gather(call, return_exceptions=True)
        if shutdown is not None:
            await shutdown
        await cleanup_fetches(data.sec.http)
        await data.aclose()


async def test_repeated_waiter_cancellation_does_not_interrupt_stream_cleanup() -> None:
    closing, release = asyncio.Event(), asyncio.Event()

    class SlowCloseStream(DripStream):
        async def aclose(self) -> None:
            closing.set()
            await release.wait()
            self.closed.set()

    stream = SlowCloseStream()
    http = make_http(httpx.MockTransport(lambda request: httpx.Response(200, stream=stream)))
    call = asyncio.create_task(
        http.get("https://data.sec.gov/x", key="k", endpoint="x", parse=parse_json)
    )
    try:
        await stream.started.wait()
        call.cancel()
        await closing.wait()
        call.cancel()
        await asyncio.sleep(0.005)
        assert not call.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert stream.closed.is_set() and not http._tasks
    finally:
        release.set()
        await asyncio.gather(call, return_exceptions=True)
        await cleanup_fetches(http)


def test_maximum_concept_and_unit_fit_external_identity_without_widening_corpus_ids() -> None:
    from dataclasses import replace

    from pydantic import ValidationError

    from ledger_rag.core.models import Citation
    from ledger_rag.external.normalize import sec_fact
    from ledger_rag.external.tools import citation_from_fact
    from tests.unit.external_fakes import NOW
    from tests.unit.test_external_sec import REVENUE_2009, pick
    from tests.unit.test_external_service import corpus_citation

    selection = pick(REVENUE_2009, "FY2009", date(2009, 12, 31))
    selection = replace(
        selection,
        fact=replace(selection.fact, unit="U" * 60),
        concept=replace(selection.concept, concept="A" * 132, taxonomy="ifrs-full"),
    )
    fact = sec_fact(selection, as_of=date(2009, 12, 31), retrieved_at=NOW)
    assert len(fact.provenance.evidence_id) == 261
    assert citation_from_fact(fact, 1).provenance == fact.provenance
    with pytest.raises(ValidationError, match="Corpus document identity"):
        Citation.model_validate({**corpus_citation(1).model_dump(), "document_id": "x" * 201})
