from datetime import date
from decimal import Decimal

import httpx
import pytest
from pydantic import SecretStr

from ledger_rag.external.errors import ExternalDataError
from ledger_rag.external.fred import (
    SeriesMeta,
    Vintage,
    fred_error,
    known_values,
    parse_observations,
    parse_series,
)
from ledger_rag.external.normalize import fred_units, fred_value
from ledger_rag.external.service import ExternalData, fred_key
from ledger_rag.generation.program import extract_numbers
from tests.unit.external_fakes import (
    FRED_KEY,
    FakeUpstream,
    day,
    external_data,
    external_settings,
    fixture,
)


def vintage(observed: str, value: str | None, start: str, end: str = "9999-12-31") -> Vintage:
    return Vintage(
        day=day(observed),
        value=None if value is None else Decimal(value),
        realtime_start=day(start),
        realtime_end=date.max if end == "9999-12-31" else day(end),
    )


def test_known_values_follow_the_real_time_period() -> None:
    rows = [
        vintage("2023-07-01", "27623.543", "2023-10-26", "2023-11-28"),
        vintage("2023-07-01", "27644.463", "2023-11-29", "2023-12-20"),
        vintage("2023-07-01", "27610.128", "2023-12-21"),
        vintage("2023-10-01", "27938.831", "2024-01-25"),
    ]
    before = known_values(rows, day("2023-10-25"))
    assert before == {}
    first = known_values(rows, day("2023-11-01"))[day("2023-07-01")]
    assert first is not None
    assert (first.value, first.vintage) == (Decimal("27623.543"), day("2023-10-26"))
    assert first.revised_after == day("2023-11-29") and not first.revised_before
    last = known_values(rows, day("2024-02-01"))
    third = last[day("2023-07-01")]
    assert third is not None and third.revised_before and third.revised_after is None
    assert day("2023-10-01") in last


def test_missing_value_sentinel_is_kept_distinct_from_unknown() -> None:
    rows = [vintage("2023-07-04", None, "2023-07-05"), vintage("2023-07-05", "3.94", "2023-07-06")]
    known = known_values(rows, day("2023-07-06"))
    assert known[day("2023-07-04")] is None
    assert day("2023-07-05") in known
    assert day("2023-07-05") not in known_values(rows, day("2023-07-05"))


def test_parsers_validate_shapes() -> None:
    meta = parse_series(fixture("fred", "series_GDP.json"))
    assert (meta.id, meta.frequency_short, meta.units) == ("GDP", "Q", "Billions of Dollars")
    assert meta.period_bounds(day("2023-07-01")) == (day("2023-07-01"), day("2023-09-30"))
    assert meta.gap_days == 92
    with pytest.raises(ValueError):
        parse_series({"seriess": [{"id": "bad id"}]})
    with pytest.raises(ValueError):
        parse_observations({"observations": [{"date": "2023-01-01", "value": "NaN"}]})
    with pytest.raises(ValueError):
        parse_observations(
            {
                "observations": [
                    {
                        "date": "2023-01-01",
                        "value": "1e999999",
                        "realtime_start": "2023-01-01",
                        "realtime_end": "9999-12-31",
                    }
                ]
            }
        )
    with pytest.raises(ExternalDataError) as caught:
        parse_observations({"count": 30000, "observations": []})
    assert caught.value.kind == "too_large"


def test_monthly_and_annual_period_bounds() -> None:
    monthly = SeriesMeta("FEDFUNDS", "t", "Monthly", "M", "Percent", "%", "NSA")
    assert monthly.period_bounds(day("2023-12-01")) == (day("2023-12-01"), day("2023-12-31"))
    annual = SeriesMeta("X", "t", "Annual", "A", "Percent", "%", "NSA")
    assert annual.period_bounds(day("2023-01-01")) == (day("2023-01-01"), day("2023-12-31"))
    daily = SeriesMeta("DFF", "t", "Daily", "D", "Percent", "%", "NSA")
    assert daily.period_bounds(day("2023-12-01")) == (None, day("2023-12-01"))


@pytest.mark.parametrize(
    ("status", "body", "kind"),
    [
        (400, "error_api_key.json", "forbidden"),
        (400, "error_no_series.json", "not_found"),
        (400, "error_not_in_alfred.json", "invalid_request"),
        (429, "error_rate_limit.json", "rate_limited"),
    ],
)
def test_fred_errors_map_to_safe_kinds(status: int, body: str, kind: str) -> None:
    error = fred_error(status, httpx.Response(200, json=fixture("fred", body)).content)
    assert error is not None and error.kind == kind
    assert "api_key" not in error.message and "http" not in error.message
    assert fred_error(500, b"{}") is None
    assert fred_error(400, b"<html>") is None


@pytest.mark.parametrize(
    ("value", "units", "text", "unit", "scale"),
    [
        ("5.33", "Percent", "5.33%", "percent", "1"),
        ("27610.128", "Billions of Dollars", "USD 27,610.128 billion", "currency", "1000000000"),
        ("-0.25", "Percentage Points", "-0.25 percentage points", "percentage_points", "1"),
        ("307.619", "Index 1982-1984=100", "307.619 (index)", None, "1"),
        ("158000", "Thousands of Persons", "158,000 thousand persons", None, "1000"),
        ("80.1", "Dollars per Barrel", "USD 80.1 per barrel", "currency", "1"),
        ("70000", "Chained 2017 Dollars", "USD 70,000", "currency", "1"),
    ],
)
def test_fred_values_render_one_extractable_number(
    value: str, units: str, text: str, unit: str | None, scale: str
) -> None:
    rendered = fred_value(Decimal(value), units)
    assert rendered == text
    mentions = extract_numbers(f"observation for 2023-10-01: {rendered}, as published")
    assert [(mention.unit, str(mention.scale)) for mention in mentions] == [(unit, scale)]
    assert fred_units("Millions of Dollars")[:2] == (6, "million")


async def test_dates_pick_the_observation_known_at_the_cutoff() -> None:
    data = external_data()
    result = await data.macro_series(
        "FEDFUNDS", as_of=day("2023-10-15"), dates=[day("2023-09-30"), day("2022-09-30")]
    )
    assert [fact.provenance.evidence_id for fact in result.facts] == [
        "fred:FEDFUNDS:2023-09-01:2023-10-02",
        "fred:FEDFUNDS:2022-09-01:2022-10-03",
    ]
    first = result.facts[0]
    assert first.quote == (
        "FRED series FEDFUNDS (Federal Funds Effective Rate), monthly observation for "
        "2023-09-01: 5.33%, as published in the vintage of 2023-10-02."
    )
    assert first.provenance.value == Decimal("5.33")
    assert (first.provenance.period_start, first.provenance.period_end) == (
        day("2023-09-01"),
        day("2023-09-30"),
    )
    assert first.provenance.source_url == "https://fred.stlouisfed.org/series/FEDFUNDS"
    assert result.subject == "FEDFUNDS: Federal Funds Effective Rate (Percent, Monthly)"
    await data.aclose()


async def test_unpublished_observations_are_missing_not_guessed() -> None:
    data = external_data()
    result = await data.macro_series("FEDFUNDS", as_of=day("2023-09-15"), dates=[day("2023-09-30")])
    assert result.facts == []
    assert [miss.payload() for miss in result.missing] == [
        {"item": "FEDFUNDS", "period": "2023-09-30", "reason": "no value known at the as-of date"}
    ]
    await data.aclose()


async def test_revisions_are_vintage_correct_and_flagged() -> None:
    data = external_data()
    second = await data.macro_series("GDP", as_of=day("2023-12-01"), dates=[day("2023-07-01")])
    fact = second.facts[0]
    assert fact.provenance.value == Decimal("27644.463")
    assert fact.provenance.published_at == day("2023-11-29")
    assert fact.provenance.freshness == "superseded"
    assert fact.provenance.superseded_by == "2023-12-21"
    assert fact.provenance.revised_before_as_of
    assert fact.provenance.scale == 9
    assert "27,610" not in fact.quote and "27,623" not in fact.quote
    final = await data.macro_series("GDP", as_of=day("2026-08-01"), dates=[day("2023-09-30")])
    assert final.facts[0].provenance.value == Decimal("27967.697")
    assert final.facts[0].provenance.freshness == "current"
    await data.aclose()


async def test_ranges_latest_and_sentinels() -> None:
    data = external_data()
    window = await data.macro_series(
        "DGS10", as_of=day("2023-07-10"), start=day("2023-07-03"), end=day("2023-07-07")
    )
    assert [fact.provenance.period_end for fact in window.facts] == [
        day("2023-07-03"),
        day("2023-07-05"),
        day("2023-07-06"),
        day("2023-07-07"),
    ]
    assert [miss.period for miss in window.missing] == ["2023-07-04"]
    assert all("10-Year" not in fact.quote for fact in window.facts)
    assert all(len(extract_numbers(fact.quote)) == 1 for fact in window.facts)
    holiday = await data.macro_series("DGS10", as_of=day("2023-07-10"), dates=[day("2023-07-04")])
    assert holiday.facts[0].provenance.period_end == day("2023-07-03")
    latest = await data.macro_series("CPIAUCSL", as_of=day("2023-11-30"))
    assert latest.facts[0].quote.endswith(
        "2023-10-01: 307.619 (index), as published in the vintage of 2023-11-14."
    )
    assert latest.facts[0].provenance.freshness == "current"
    stale = await data.macro_series("CPIAUCSL", as_of=day("2024-03-01"))
    assert stale.facts == [] and stale.missing[0].period == "latest"
    with pytest.raises(ExternalDataError, match="at most 8 per call"):
        await data.macro_series(
            "FEDFUNDS", as_of=day("2023-12-31"), start=day("2023-01-01"), end=day("2023-12-31")
        )
    with pytest.raises(ExternalDataError, match="ten years"):
        await data.macro_series(
            "FEDFUNDS", as_of=day("2023-12-31"), start=day("2000-01-01"), end=day("2023-12-31")
        )
    with pytest.raises(ExternalDataError, match="No observations can be known"):
        await data.macro_series("FEDFUNDS", as_of=day("2023-01-01"), start=day("2023-06-01"))
    await data.aclose()


async def test_latest_is_stale_when_the_cadence_lapses() -> None:
    data = external_data()
    result = await data.macro_series("GDP", as_of=day("2024-06-20"))
    fact = result.facts[0]
    assert fact.provenance.period_label == "2023-10-01"
    assert fact.provenance.freshness == "stale"
    await data.aclose()


@pytest.mark.parametrize(
    ("series", "kind"),
    [("NOALFRED", "invalid_request"), ("NOPE", "not_found"), ("bad id!", "invalid_request")],
)
async def test_series_errors(series: str, kind: str) -> None:
    data = external_data()
    with pytest.raises(ExternalDataError) as caught:
        await data.macro_series(series, as_of=day("2023-11-30"))
    assert caught.value.kind == kind
    assert data.status().fred.enabled
    await data.aclose()


def test_fred_key_states() -> None:
    assert fred_key(external_settings(fred_api_key=None))[0] is None
    assert "LEDGER_FRED_API_KEY" in str(fred_key(external_settings(fred_api_key=None))[1])
    key, reason = fred_key(external_settings(fred_api_key=SecretStr(f" {FRED_KEY}\n")))
    assert (key, reason) == (FRED_KEY, None)
    bad, why = fred_key(external_settings(fred_api_key=SecretStr("not-a-key")))
    assert bad is None and why is not None and "not-a-key" not in why


async def test_missing_key_disables_fred_with_a_reason_and_no_requests() -> None:
    upstream = FakeUpstream()
    data = ExternalData(external_settings(fred_api_key=None), transport=upstream.transport)
    status = data.status()
    assert status.enabled and status.sec.enabled
    assert not status.fred.enabled and "no FRED API key" in str(status.fred.reason)
    with pytest.raises(ExternalDataError) as caught:
        await data.macro_series("FEDFUNDS", as_of=day("2023-10-15"))
    assert caught.value.kind == "disabled"
    assert upstream.requests == []
    await data.aclose()


async def test_rejected_key_disables_fred_until_restart() -> None:
    upstream = FakeUpstream()
    data = external_data(upstream, fred_api_key=SecretStr("f" * 32))
    with pytest.raises(ExternalDataError) as caught:
        await data.macro_series("FEDFUNDS", as_of=day("2023-10-15"))
    assert caught.value.kind == "forbidden"
    assert data.status().fred.reason == "FRED rejected the configured API key"
    assert data.status().fred.circuit == "closed"
    with pytest.raises(ExternalDataError) as again:
        await data.macro_series("FEDFUNDS", as_of=day("2023-10-15"))
    assert again.value.kind == "disabled"
    assert len(upstream.requests) == 1
    assert all("f" * 32 not in repr(record) for record in [caught.value, again.value])
    await data.aclose()


async def test_key_is_sent_only_as_a_query_parameter_to_fred() -> None:
    upstream = FakeUpstream()
    data = external_data(upstream)
    await data.macro_series("FEDFUNDS", as_of=day("2023-10-15"))
    await data.company_facts("AAPL", ["InterestExpense"], ["FY2023"], as_of=day("2026-08-01"))
    fred = [request for request in upstream.requests if request.url.host == "api.stlouisfed.org"]
    sec = [request for request in upstream.requests if request.url.host != "api.stlouisfed.org"]
    assert fred and all(request.url.params["api_key"] == FRED_KEY for request in fred)
    assert all(request.url.params["file_type"] == "json" for request in fred)
    assert all(FRED_KEY not in str(request.url) for request in sec)
    observations = [request for request in fred if request.url.path.endswith("/observations")]
    assert observations[0].url.params["realtime_start"] == "1776-07-04"
    assert observations[0].url.params["realtime_end"] == "9999-12-31"
    await data.aclose()
