"""Opt-in checks against the real SEC EDGAR and FRED APIs (``LEDGER_LIVE_TESTS=1``).

Each test makes a handful of polite requests (declared User-Agent, shared rate limiter,
response cache) and asserts values traceable to specific filings and vintages.
"""

import os
from datetime import date
from decimal import Decimal

import pytest
from pydantic import SecretStr

from ledger_rag.core.config import Settings
from ledger_rag.external.service import ExternalData

pytestmark = pytest.mark.live


@pytest.fixture(autouse=True)
def no_retry_sleep() -> None:
    """Live runs keep real backoff so a 429 or 503 is honoured, not hammered."""


def live_settings() -> Settings:
    return Settings(
        demo_token=SecretStr("live-test-token"),
        external_data_enabled=True,
        sec_max_requests_per_second=2,
    )


async def test_sec_point_in_time_facts() -> None:
    data = ExternalData(live_settings())
    try:
        current = await data.company_facts(
            "AAPL", ["InterestExpense"], ["FY2023"], as_of=date(2024, 6, 30)
        )
        fact = current.facts[0].provenance
        assert (fact.value, fact.identifier, fact.form) == (
            Decimal("3933000000"),
            "0000320193-23-000106",
            "10-K",
        )
        assert fact.published_at == date(2023, 11, 3)
        before = await data.company_facts(
            "AAPL", ["InterestExpense"], ["FY2022", "FY2023"], as_of=date(2023, 10, 15)
        )
        assert [item.provenance.identifier for item in before.facts] == ["0000320193-22-000108"]
        assert before.facts[0].provenance.value == Decimal("2931000000")
        assert [miss.period for miss in before.missing] == ["FY2023"]
        again = await data.read(fact.evidence_id, as_of=date(2024, 6, 30))
        assert again.quote == current.facts[0].quote
        shell = await data.company_facts("SHEL", ["Revenues"], ["FY2023"], as_of=date(2024, 6, 1))
        revenue = shell.facts[0].provenance
        assert revenue.concept == "ifrs-full:Revenue"
        assert (revenue.value, revenue.identifier) == (
            Decimal("316620000000"),
            "0001306965-24-000026",
        )
    finally:
        await data.aclose()


@pytest.mark.skipif(
    not os.environ.get("LEDGER_FRED_API_KEY"), reason="LEDGER_FRED_API_KEY is not set"
)
async def test_fred_vintage_correct_observations() -> None:
    data = ExternalData(live_settings())
    try:
        assert data.status().fred.enabled, data.status().fred.reason
        result = await data.macro_series(
            "FEDFUNDS", as_of=date(2023, 10, 15), dates=[date(2023, 9, 30)]
        )
        fact = result.facts[0].provenance
        assert fact.value == Decimal("5.33")
        assert date(2023, 10, 1) <= fact.published_at <= date(2023, 10, 15)
        unknown = await data.macro_series(
            "FEDFUNDS", as_of=date(2023, 9, 15), dates=[date(2023, 9, 30)]
        )
        assert unknown.facts == []
        gdp = await data.macro_series("GDP", as_of=date(2023, 11, 1), dates=[date(2023, 7, 1)])
        first = gdp.facts[0].provenance
        assert first.published_at <= date(2023, 11, 1)
        assert first.freshness == "superseded" and first.superseded_by is not None
    finally:
        await data.aclose()
