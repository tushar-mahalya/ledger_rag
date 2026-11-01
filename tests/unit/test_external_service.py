import asyncio
from datetime import date
from decimal import Decimal
from typing import Any

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from ledger_rag.core.config import Settings
from ledger_rag.core.models import Activity, Calculation, Citation, Principal
from ledger_rag.core.provenance import Provenance, is_source_url
from ledger_rag.external.errors import ExternalDataError
from ledger_rag.external.service import ExternalData, ExternalSession
from ledger_rag.external.tools import (
    MAX_CITATIONS,
    is_external_tool,
    run_external_tool,
    tool_schemas,
)
from ledger_rag.generation.program import run_program
from ledger_rag.generation.verification import NumericPolicy, check_numbers
from tests.unit.external_fakes import (
    NOW,
    FakeUpstream,
    day,
    external_data,
    external_settings,
)

READER = Principal(subject="analyst", tenant_id="demo", roles=["reader"], groups=["finance"])


def session(data: ExternalData, as_of: str = "2026-08-01", **changes: Any) -> ExternalSession:
    principal = changes.pop("principal", READER)
    return ExternalSession(
        data=data,
        principal=principal,
        as_of=data.as_of(day(as_of)),
        providers=data.providers(principal),
        **changes,
    )


def corpus_citation(index: int) -> Citation:
    return Citation(
        id=f"C{index}",
        document_id=f"doc-{index}",
        source="10-K",
        title="Corpus passage",
        published_at=date(2023, 11, 3),
        quote="Total net sales were $383,285 million.",
    )


def test_disabled_by_default_and_settings_fail_closed() -> None:
    default = Settings(demo_token=SecretStr("test-only-token"))
    assert default.external_data_enabled is False and default.mcp_enabled is False
    assert ExternalData.from_settings(default) is None
    for agent in ["", "Ledger RAG", " Ledger a@b.co", "Ledger\tRAG a@b.co", "Ledger RAG a@b"]:
        with pytest.raises(ValidationError, match="contact e-mail"):
            external_settings(sec_user_agent=agent)
    assert external_settings(sec_user_agent="", sec_enabled=False).sec_enabled is False
    with pytest.raises(ValidationError):
        external_settings(sec_max_requests_per_second=11)
    with pytest.raises(ValidationError):
        external_settings(fred_max_requests_per_minute=121)
    with pytest.raises(ValidationError, match="allowlist"):
        external_settings(external_allowed_tenants=[""])


async def test_status_reports_each_provider() -> None:
    data = external_data(sec_enabled=False)
    status = data.status()
    assert status.enabled and status.fred.enabled and status.fred.circuit == "closed"
    assert not status.sec.enabled and status.sec.reason == "disabled by LEDGER_SEC_ENABLED"
    assert data.providers(READER) == ["fred"]
    off = external_data(fred_enabled=False).status().fred
    assert off.reason == "disabled by LEDGER_FRED_ENABLED" and off.circuit is None
    malformed = external_data(fred_api_key=SecretStr("abc")).status().fred
    assert not malformed.enabled and "malformed" in str(malformed.reason)
    await data.aclose()


@pytest.mark.parametrize(
    ("principal", "overrides", "expected"),
    [
        (READER, {}, ["sec", "fred"]),
        (None, {}, []),
        (Principal(subject="x", tenant_id="demo", roles=["auditor"]), {}, []),
        (READER, {"external_required_group": "macro"}, []),
        (READER, {"external_required_group": "finance"}, ["sec", "fred"]),
        (READER, {"external_allowed_tenants": ["acme"]}, []),
        (READER, {"external_allowed_tenants": ["acme", "demo"]}, ["sec", "fred"]),
    ],
)
async def test_entitlement(
    principal: Principal | None, overrides: dict[str, Any], expected: list[str]
) -> None:
    data = external_data(**overrides)
    assert data.providers(principal) == expected
    await data.aclose()


def test_as_of_never_exceeds_today() -> None:
    data = external_data()
    assert data.as_of(None) == NOW.date()
    assert data.as_of(date(2030, 1, 1)) == NOW.date()
    assert data.as_of(date(2020, 1, 1)) == date(2020, 1, 1)


async def test_company_facts_combines_facts_and_misses() -> None:
    data = external_data()
    result = await data.company_facts(
        "aapl",
        ["InterestExpense", "us-gaap:InterestExpence"],
        ["FY2023", "FY2023Q4"],
        as_of=day("2026-08-01"),
    )
    assert result.subject == "Apple Inc. (CIK 0000320193, AAPL)"
    assert [fact.provenance.value for fact in result.facts] == [Decimal("3933000000")]
    reasons = {(miss.item, miss.period): miss.reason for miss in result.missing}
    assert "no discrete fiscal Q4" in reasons[("InterestExpense", "FY2023Q4")]
    assert "us-gaap:InterestExpense" in reasons[("us-gaap:InterestExpence", "FY2023")]
    with pytest.raises(ExternalDataError, match="At most 8 concept-period pairs"):
        await data.company_facts(
            "AAPL", ["A", "B", "C"], ["FY2021", "FY2022", "FY2023"], as_of=NOW.date()
        )
    with pytest.raises(ExternalDataError) as unknown:
        await data.company_facts("Nonexistent Holdings", ["Assets"], ["FY2023"], as_of=NOW.date())
    assert unknown.value.kind == "not_found"
    await data.aclose()


async def test_publication_cutoff_hides_later_filings() -> None:
    data = external_data()
    early = await data.company_facts(
        "AAPL", ["InterestExpense"], ["FY2023"], as_of=day("2023-11-02")
    )
    assert early.facts == [] and "had been filed as of 2023-11-02" in early.missing[0].reason
    filed = await data.company_facts(
        "AAPL", ["InterestExpense"], ["FY2023"], as_of=day("2023-11-03")
    )
    assert filed.facts[0].provenance.published_at == day("2023-11-03")
    await data.aclose()


async def test_breaker_opens_after_provider_failures_and_bad_requests_do_not_count() -> None:
    upstream = FakeUpstream()
    upstream.overrides["CIK0000320193"] = lambda request: httpx.Response(503)
    data = external_data(upstream, external_breaker_failures=2, retry_attempts=1)
    for _ in range(4):
        with pytest.raises(ExternalDataError) as bad:
            await data.company_facts("AAPL", ["Assets"], ["FY2023-13"], as_of=NOW.date())
        assert bad.value.kind == "invalid_request"
    assert data.status().sec.circuit == "closed"
    for _ in range(2):
        with pytest.raises(ExternalDataError) as failed:
            await data.company_facts("AAPL", ["Assets"], ["FY2023"], as_of=NOW.date())
        assert failed.value.kind == "upstream"
    assert data.status().sec.circuit == "open"
    calls = len(upstream.requests)
    with pytest.raises(ExternalDataError) as blocked:
        await data.company_facts("AAPL", ["Assets"], ["FY2023"], as_of=NOW.date())
    assert blocked.value.kind == "circuit_open" and len(upstream.requests) == calls
    assert data.status().fred.circuit == "closed"
    await data.aclose()


async def test_whole_lookup_times_out_and_counts_as_a_failure() -> None:
    release = asyncio.Event()

    class SlowTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            await release.wait()
            return httpx.Response(200, json={})

    data = ExternalData(
        external_settings(external_timeout_seconds=0.05, external_breaker_failures=1),
        transport=SlowTransport(),
        now=lambda: NOW,
    )
    with pytest.raises(ExternalDataError) as caught:
        await data.company_facts("AAPL", ["Assets"], ["FY2023"], as_of=NOW.date())
    assert caught.value.kind == "timeout"
    assert str(caught.value) == "SEC EDGAR did not respond in time"
    assert data.status().sec.circuit == "open"
    release.set()
    await data.aclose()


async def test_read_round_trips_evidence_ids_under_the_cutoff() -> None:
    data = external_data()
    found = await data.company_facts(
        "AAPL", ["InterestExpense", "Assets"], ["FY2023"], as_of=day("2026-08-01")
    )
    for fact in found.facts:
        again = await data.read(fact.provenance.evidence_id, as_of=day("2026-08-01"))
        assert again.quote == fact.quote
        assert again.provenance.value == fact.provenance.value
        assert again.provenance.evidence_id == fact.provenance.evidence_id
    evidence_id = found.facts[0].provenance.evidence_id
    for as_of in ["2023-11-02", "2020-01-01"]:
        with pytest.raises(ExternalDataError, match="^Evidence not found$"):
            await data.read(evidence_id, as_of=day(as_of))
    for forged in [
        evidence_id.replace("0000320193-23-000106", "0000320193-23-000107"),
        evidence_id.replace("2023-09-30", "2023-09-29"),
        "sec:0000320193:us-gaap:NoSuchConcept:0000320193-23-000106:2023-09-30",
        "sec:320193:us-gaap:InterestExpense:0000320193-23-000106:2023-09-30",
        "corpus:doc-1",
        "fred:fedfunds:2023-09-01:2023-10-02",
        "fred:FEDFUNDS:2023-09-01:2023-10-03",
    ]:
        with pytest.raises(ExternalDataError, match="^Evidence not found$"):
            await data.read(forged, as_of=day("2026-08-01"))
    fred = await data.read("fred:FEDFUNDS:2023-09-01:2023-10-02", as_of=day("2023-10-15"))
    assert fred.provenance.value == Decimal("5.33")
    with pytest.raises(ExternalDataError, match="^Evidence not found$"):
        await data.read("fred:FEDFUNDS:2023-09-01:2023-10-02", as_of=day("2023-10-01"))
    await data.aclose()


async def test_read_restated_sec_fact_reports_supersession() -> None:
    data = external_data()
    original = await data.company_facts(
        "AAPL", ["SalesRevenueNet"], ["FY2009"], as_of=day("2009-12-31")
    )
    fact = original.facts[0]
    again = await data.read(fact.provenance.evidence_id, as_of=day("2026-08-01"))
    assert again.provenance.value == Decimal("36537000000")
    assert again.provenance.freshness == "superseded"
    assert again.provenance.superseded_by == "0001193125-10-012091"
    await data.aclose()


async def test_session_budget_is_per_turn() -> None:
    data = external_data(external_max_calls_per_turn=1)
    turn = session(data)
    turn.spend()
    with pytest.raises(ExternalDataError) as caught:
        turn.spend()
    assert caught.value.kind == "budget_exhausted"
    session(data).spend()
    await data.aclose()


async def test_tool_schemas_follow_provider_availability() -> None:
    data = external_data(fred_api_key=None)
    names = [schema["function"]["name"] for schema in tool_schemas(session(data))]
    assert names == ["lookup_company_facts"]
    assert tool_schemas(None) == []
    outsider = Principal(subject="x", tenant_id="demo", roles=["guest"])
    assert tool_schemas(session(data, principal=outsider)) == []
    assert is_external_tool("lookup_macro_series") and not is_external_tool("calculator")
    await data.aclose()


async def test_run_tool_appends_citations_and_deduplicates() -> None:
    data = external_data()
    turn = session(data)
    citations = [corpus_citation(1)]
    evidence_ids = {"doc-1"}
    arguments = {
        "company": "AAPL",
        "concepts": ["InterestExpense"],
        "periods": ["FY2022", "FY2023"],
    }
    result = await run_external_tool(
        "lookup_company_facts", arguments, turn, citations, evidence_ids
    )
    assert [item["id"] for item in result["evidence"]] == ["C2", "C3"]
    assert result["as_of"] == "2026-08-01" and "missing" not in result
    assert result["evidence"][0]["text"] == citations[1].quote
    assert result["evidence"][0]["source_kind"] == "sec_xbrl"
    assert citations[1].document_id in evidence_ids
    assert citations[1].provenance is not None and citations[1].source_kind == "sec_xbrl"
    again = await run_external_tool(
        "lookup_company_facts",
        {"company": "320193", "concepts": ["InterestExpense"], "periods": ["FY2023"]},
        turn,
        citations,
        evidence_ids,
    )
    assert [item["id"] for item in again["evidence"]] == ["C3"] and len(citations) == 3
    with pytest.raises(ExternalDataError) as spent:
        await run_external_tool("lookup_company_facts", arguments, turn, citations, evidence_ids)
    assert spent.value.kind == "budget_exhausted"
    await data.aclose()


async def test_run_tool_respects_the_citation_cap() -> None:
    data = external_data()
    citations = [corpus_citation(index) for index in range(1, MAX_CITATIONS)]
    result = await run_external_tool(
        "lookup_company_facts",
        {"company": "AAPL", "concepts": ["InterestExpense"], "periods": ["FY2022", "FY2023"]},
        session(data),
        citations,
        set(),
    )
    assert len(citations) == MAX_CITATIONS and len(result["evidence"]) == 1
    assert "truncated" in result
    with pytest.raises(ExternalDataError, match="evidence limit"):
        await run_external_tool(
            "lookup_macro_series", {"series_id": "FEDFUNDS"}, session(data), citations, set()
        )
    await data.aclose()


@pytest.mark.parametrize(
    ("name", "arguments", "fragment"),
    [
        ("lookup_company_facts", {"company": "AAPL"}, "concepts"),
        ("lookup_company_facts", {"company": "AAPL", "concepts": ["x" * 143]}, "142"),
        ("lookup_company_facts", {"company": "AAPL", "concepts": ["A"], "sql": "1"}, "sql"),
        ("lookup_company_facts", {"company": "", "concepts": ["A"]}, "company"),
        (
            "lookup_macro_series",
            {"series_id": "GDP", "start": "2023-02-01", "end": "2023-01-01"},
            "start",
        ),
        ("lookup_macro_series", {"series_id": "GDP", "end": "2023-01-01"}, "end requires start"),
        (
            "lookup_macro_series",
            {"series_id": "GDP", "dates": ["2023-01-01"], "start": "2023-01-01"},
            "either",
        ),
        ("lookup_macro_series", {"series_id": "GDP", "dates": ["yesterday<script>"]}, "dates"),
    ],
)
async def test_invalid_arguments_are_sanitized_and_free(
    name: Any, arguments: dict[str, Any], fragment: str
) -> None:
    upstream = FakeUpstream()
    data = external_data(upstream)
    turn = session(data)
    with pytest.raises(ExternalDataError) as caught:
        await run_external_tool(name, arguments, turn, [], set())
    assert caught.value.kind == "invalid_request" and fragment in caught.value.message
    assert "<script>" not in caught.value.message
    assert turn.calls == 0 and upstream.requests == []
    await data.aclose()


async def test_disabled_provider_tool_is_refused_without_spending() -> None:
    data = external_data(fred_api_key=None)
    turn = session(data)
    with pytest.raises(ExternalDataError) as caught:
        await run_external_tool("lookup_macro_series", {"series_id": "GDP"}, turn, [], set())
    assert caught.value.kind == "disabled" and turn.calls == 0
    await data.aclose()


def test_citation_provenance_must_match_its_kind(settings: Settings) -> None:
    base = corpus_citation(1).model_dump()
    with pytest.raises(ValidationError):
        Citation(**{**base, "source_kind": "sec_xbrl"})
    provenance = Provenance(
        source_kind="fred",
        provider="FRED",
        evidence_id="fred:GDP:2023-07-01:2023-11-29",
        identifier="GDP",
        entity="Federal Reserve Bank of St. Louis (FRED/ALFRED)",
        concept="Gross Domestic Product",
        period_label="2023-07-01",
        period_start=date(2023, 7, 1),
        period_end=date(2023, 9, 30),
        unit="Billions of Dollars",
        scale=9,
        value=Decimal("27644.463"),
        published_at=date(2023, 11, 29),
        retrieved_at=NOW,
        as_of=date(2023, 12, 1),
        source_url="https://fred.stlouisfed.org/series/GDP",
        freshness="superseded",
    )
    with pytest.raises(ValidationError):
        Citation(**{**base, "provenance": provenance})
    with pytest.raises(ValidationError):
        Citation(**{**base, "source_kind": "sec_xbrl", "provenance": provenance})
    assert Citation(**{**base, "source_kind": "fred", "provenance": provenance}).provenance
    with pytest.raises(ValidationError):
        provenance.model_copy(update={}).model_validate(
            {**provenance.model_dump(), "source_url": "https://evil.example/sec.gov"}
        )
    with pytest.raises(ValidationError):
        Provenance.model_validate({**provenance.model_dump(), "value": Decimal("1e40")})
    assert not is_source_url("http://www.sec.gov/x")
    assert not is_source_url("https://www.sec.gov.evil.example/x")
    assert not is_source_url("https://user@www.sec.gov/x")
    assert not is_source_url("https://data.sec.gov/api/xbrl/x.json")
    assert is_source_url("https://www.sec.gov/Archives/edgar/data/320193/x-index.htm")


async def test_external_quotes_drive_programs_and_numeric_verification() -> None:
    data = external_data()
    turn = session(data, as_of="2023-11-30")
    citations: list[Citation] = []
    await run_external_tool(
        "lookup_company_facts",
        {"company": "AAPL", "concepts": ["InterestExpense"], "periods": ["FY2022", "FY2023"]},
        turn,
        citations,
        set(),
    )
    await run_external_tool(
        "lookup_macro_series",
        {"series_id": "FEDFUNDS", "dates": ["2022-09-30", "2023-09-30"]},
        turn,
        citations,
        set(),
    )
    assert [citation.source_kind for citation in citations] == ["sec_xbrl"] * 2 + ["fred"] * 2
    calculation: Calculation = run_program(
        {
            "steps": [
                {
                    "op": "percent_change",
                    "args": [
                        {"citation_id": "C1", "quote": "USD 2,931,000,000"},
                        {"citation_id": "C2", "quote": "USD 3,933,000,000"},
                    ],
                }
            ]
        },
        citations,
    )
    assert Decimal(calculation.result).quantize(Decimal("0.01")) == Decimal("34.19")
    answer = (
        "Apple's interest expense rose from $2.931 billion in FY2022 to $3.933 billion in "
        "FY2023 [C1][C2], up 34.19%, while the effective federal funds rate climbed from "
        "2.56% to 5.33% [C3][C4]."
    )
    check = check_numbers(answer, citations, [calculation], NumericPolicy())
    assert check.unsupported == [] and check.checked == 5
    wrong = answer.replace("5.33%", "5.50%")
    assert check_numbers(wrong, citations, [calculation], NumericPolicy()).unsupported == ["5.50%"]
    await data.aclose()


def test_external_activity_outcome_stays_off_the_wire_until_set() -> None:
    started = Activity(
        stage="tool", status="started", label="Looking up SEC filings", tool="lookup_company_facts"
    )
    assert "outcome" not in started.model_dump(exclude_none=True)
    done = started.model_copy(update={"status": "completed", "outcome": "timeout"})
    assert done.model_dump(exclude_none=True)["outcome"] == "timeout"
    with pytest.raises(ValidationError):
        Activity(stage="tool", status="completed", label="Looking up macro data", outcome="slow")
