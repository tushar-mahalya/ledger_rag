from datetime import date
from decimal import Decimal

import httpx
import pytest

from ledger_rag.external.errors import ExternalDataError
from ledger_rag.external.http import parse_json
from ledger_rag.external.normalize import sec_fact, sec_value
from ledger_rag.external.sec import (
    MAX_CONCEPT_LENGTH,
    CompanyIndex,
    ConceptFacts,
    PeriodSpec,
    Selection,
    SelectionMiss,
    concept_candidates,
    filing_url,
    normalize_name,
    normalize_ticker,
    parse_cik,
    parse_concept,
    parse_concept_index,
    parse_period,
    select_fact,
    unit_order,
)
from ledger_rag.external.tools import citation_from_fact
from ledger_rag.generation.program import extract_numbers, run_program
from tests.unit.external_fakes import FIXTURES, NOW, FakeUpstream, external_data

AS_OF = date(2026, 8, 1)


def concept(name: str) -> ConceptFacts:
    return parse_concept(parse_json((FIXTURES / "sec" / f"concept_{name}.json").read_bytes()))


def pick(name: str, period: str, as_of: date = AS_OF, unit: str | None = None) -> Selection:
    return select_fact(concept(name), parse_period(period), as_of, unit)


INTEREST = "0000320193_us-gaap_InterestExpense"
REVENUE_2009 = "0000320193_us-gaap_SalesRevenueNet"
ASSETS = "0000320193_us-gaap_Assets"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("320193", 320193),
        ("0000320193", 320193),
        ("CIK0000320193", 320193),
        ("cik 320193", 320193),
        ("0", None),
        ("12345678901", None),
        ("AAPL", None),
    ],
)
def test_cik_parsing_and_padding(text: str, expected: int | None) -> None:
    assert parse_cik(text) == expected


def test_ticker_and_name_normalization() -> None:
    assert normalize_ticker("brk.b") == normalize_ticker("BRK/B") == "BRK-B"
    assert normalize_ticker("BRK B") == "BRK-B"
    assert normalize_ticker("$$$") is None
    assert normalize_name("The Coca-Cola Company") == normalize_name("COCA COLA CO") == "coca cola"


def test_company_index_maps_share_classes_to_one_cik() -> None:
    index = CompanyIndex.from_payload(
        parse_json((FIXTURES / "sec" / "company_tickers.json").read_bytes())
    )
    assert index.by_ticker["GOOG"] is index.by_ticker["GOOGL"]
    assert index.by_ticker["BRK-A"].cik == index.by_ticker["BRK-B"].cik == 1067983
    assert index.by_cik[320193].padded == "0000320193"
    assert set(index.by_cik[1652044].tickers) >= {"GOOGL", "GOOG"}
    with pytest.raises(ValueError):
        CompanyIndex.from_payload({"0": {"cik_str": "x"}})


@pytest.mark.parametrize(
    ("text", "spec"),
    [
        ("FY2023", PeriodSpec("fy", year=2023)),
        ("fy 2023", PeriodSpec("fy", year=2023)),
        ("FY2023Q2", PeriodSpec("fq", year=2023, quarter=2)),
        ("Q2 FY2023", PeriodSpec("fq", year=2023, quarter=2)),
        ("CY2023", PeriodSpec("cy", year=2023)),
        ("CY2023Q4", PeriodSpec("cy", year=2023, quarter=4)),
        ("CY2023Q4I", PeriodSpec("cy", year=2023, quarter=4, instant=True)),
        ("2023-09-30", PeriodSpec("date", day=date(2023, 9, 30))),
        ("latest", PeriodSpec("latest")),
    ],
)
def test_period_grammar(text: str, spec: PeriodSpec) -> None:
    assert parse_period(text) == spec


@pytest.mark.parametrize("text", ["2023", "FY23", "CY2023I", "FY2023Q5", "2023-02-30", "FY1900"])
def test_period_grammar_rejects_ambiguous_input(text: str) -> None:
    with pytest.raises(ExternalDataError) as caught:
        parse_period(text)
    assert caught.value.kind == "invalid_request"


def test_concept_candidates_fall_back_from_us_gaap_to_ifrs() -> None:
    assert concept_candidates("Revenues") == [
        ("us-gaap", "Revenues"),
        ("ifrs-full", "Revenues"),
        ("ifrs-full", "Revenue"),
        ("dei", "Revenues"),
    ]
    assert concept_candidates("ifrs-full:Revenue") == [("ifrs-full", "Revenue")]
    with pytest.raises(ExternalDataError):
        concept_candidates("us-gaap:../../etc")
    longest = "A" * MAX_CONCEPT_LENGTH
    assert concept_candidates(f"ifrs-full:{longest}") == [("ifrs-full", longest)]
    with pytest.raises(ExternalDataError, match="at most 132 characters"):
        concept_candidates(longest + "B")
    worst = f"sec:0000320193:ifrs-full:{longest}:0000320193-23-000106:2022-09-25_2023-09-30"
    assert len(worst) == 200


def test_parse_concept_skips_malformed_facts() -> None:
    parsed = parse_concept(
        {
            "cik": 1,
            "taxonomy": "us-gaap",
            "tag": "Assets",
            "label": None,
            "entityName": " ",
            "units": {
                "USD": [
                    {"end": "2023-12-31", "val": 1.5, "accn": "0000000001-24-000001"},
                    {"end": "2023-12-31", "val": 1, "accn": "bad", "form": "10-K"},
                    {"end": "2023-12-31", "val": True, "accn": "0000000001-24-000001"},
                    {
                        "start": "2024-01-02",
                        "end": "2023-12-31",
                        "val": 1,
                        "accn": "0000000001-24-000001",
                        "form": "10-K",
                        "filed": "2024-02-01",
                    },
                    {
                        "end": "2023-12-31",
                        "val": Decimal("7"),
                        "accn": "0000000001-24-000001",
                        "form": "10-K",
                        "filed": "2024-02-01",
                        "fy": 2023,
                        "fp": "FY",
                    },
                ]
            },
        }
    )
    assert [fact.value for fact in parsed.facts] == [Decimal("7")]
    assert parsed.label is None and parsed.entity == "CIK 1"
    with pytest.raises(ValueError):
        parse_concept({"units": {}, "cik": 1, "taxonomy": "custom", "tag": "X"})


def test_fiscal_year_uses_only_filings_known_at_the_cutoff() -> None:
    selection = pick(INTEREST, "FY2023")
    assert selection.fact.value == Decimal("3933000000")
    assert (selection.fact.accn, selection.fact.form) == ("0000320193-23-000106", "10-K")
    assert selection.fact.period == (date(2022, 9, 25), date(2023, 9, 30))
    assert not selection.restated and selection.superseded_by is None
    with pytest.raises(SelectionMiss, match="no FY2023 annual report had been filed as of"):
        pick(INTEREST, "FY2023", date(2023, 11, 2))
    on_filing_day = pick(INTEREST, "FY2023", date(2023, 11, 3))
    assert on_filing_day.fact.accn == "0000320193-23-000106"


def test_prior_year_prefers_the_latest_filing_known_at_the_cutoff() -> None:
    assert pick(INTEREST, "FY2022").fact.accn == "0000320193-23-000106"
    assert pick(INTEREST, "FY2022", date(2023, 10, 15)).fact.accn == "0000320193-22-000108"


def test_restatement_is_exposed_before_and_after_the_amendment() -> None:
    before = pick(REVENUE_2009, "FY2009", date(2009, 12, 31))
    assert before.fact.value == Decimal("36537000000")
    assert before.fact.form == "10-K"
    assert before.superseded_by is not None
    assert before.superseded_by.accn == "0001193125-10-012091"
    assert not before.restated

    after = pick(REVENUE_2009, "FY2009", date(2010, 6, 30))
    assert after.fact.value == Decimal("42905000000")
    assert (after.fact.form, after.fact.amended) == ("10-K/A", True)
    assert after.restated and after.superseded_by is None


def test_same_day_amendment_wins_the_tie() -> None:
    selection = pick(ASSETS, "2009-09-26", date(2010, 1, 25))
    assert selection.fact.value == Decimal("47501000000")
    assert selection.fact.accn == "0001193125-10-012091"
    assert selection.restated


def test_quarters_frames_and_latest() -> None:
    quarter = pick(INTEREST, "FY2023Q2")
    assert quarter.fact.value == Decimal("930000000")
    assert quarter.fact.form == "10-Q"
    assert quarter.fact.days in range(80, 101)
    with pytest.raises(SelectionMiss, match="no discrete fiscal Q4 value"):
        pick(INTEREST, "FY2023Q4")
    framed = pick(INTEREST, "CY2022")
    assert framed.fact.period == (date(2021, 9, 26), date(2022, 9, 24))
    latest = pick(INTEREST, "latest")
    assert latest.fact.value == Decimal("3933000000") and latest.stale
    earlier = pick(INTEREST, "latest", date(2023, 10, 15))
    assert earlier.fact.value == Decimal("2931000000") and not earlier.stale


def test_instant_and_duration_specs_are_not_mixed() -> None:
    with pytest.raises(SelectionMiss, match="measures a period"):
        pick(INTEREST, "2023-09-30")
    with pytest.raises(SelectionMiss, match="duration concept"):
        pick(INTEREST, "CY2023Q4I")
    shares = "0000320193_dei_EntityCommonStockSharesOutstanding"
    assert pick(shares, "CY2023").period_label == "CY2023Q4I"
    assert pick(shares, "FY2023").fact.end == date(2023, 10, 20)


def test_share_class_conflicts_are_refused() -> None:
    with pytest.raises(SelectionMiss, match="one per share class"):
        pick("0001067983_us-gaap_EarningsPerShareBasic", "FY2023")


def test_units_are_ordered_and_requestable() -> None:
    assert unit_order(["shares", "pure", "USD/shares", "EUR", "USD"], None) == [
        "USD",
        "EUR",
        "USD/shares",
        "shares",
        "pure",
    ]
    assert unit_order(["USD", "USD/shares"], "usd-per-shares") == ["USD/shares"]
    with pytest.raises(SelectionMiss, match="no values in unit USD/shares"):
        pick(INTEREST, "FY2023", unit="USD/shares")


def test_ifrs_filers_use_20f_and_6k() -> None:
    annual = pick("0001306965_ifrs-full_Revenue", "FY2023", date(2024, 6, 1))
    assert (annual.fact.form, annual.fact.value) == ("20-F", Decimal("316620000000"))
    interim = pick("0001306965_ifrs-full_Revenue", "FY2024Q2")
    assert (interim.fact.form, interim.fact.value) == ("6-K", Decimal("74463000000"))


@pytest.mark.parametrize(
    ("value", "unit", "text"),
    [
        (Decimal("3933000000"), "USD", "USD 3,933,000,000"),
        (Decimal("-12.5"), "EUR", "EUR -12.5"),
        (Decimal("6.13"), "USD/shares", "USD 6.13 per share"),
        (Decimal("15552752000"), "shares", "15,552,752,000 shares"),
        (Decimal("0.147"), "pure", "14.7%"),
        (Decimal("1"), "pure", "100%"),
        (Decimal("5"), "Rate1", "5"),
    ],
)
def test_sec_values_render_exactly(value: Decimal, unit: str, text: str) -> None:
    assert sec_value(value, unit) == text


@pytest.mark.parametrize(
    ("name", "period", "unit", "kind"),
    [
        (INTEREST, "FY2023", "currency", "3933000000"),
        ("0000320193_us-gaap_EarningsPerShareDiluted", "FY2023", "currency", "6.13"),
        (
            "0000320193_us-gaap_EffectiveIncomeTaxRateContinuingOperations",
            "FY2023",
            "percent",
            "14.7",
        ),
        ("0000320193_dei_EntityCommonStockSharesOutstanding", "FY2023", None, "15552752000"),
    ],
)
def test_quotes_contain_exactly_one_extractable_number(
    name: str, period: str, unit: str | None, kind: str
) -> None:
    fact = sec_fact(pick(name, period), as_of=AS_OF, retrieved_at=NOW)
    mentions = extract_numbers(fact.quote)
    assert [(str(mention.value), mention.unit) for mention in mentions] == [(kind, unit)]
    # Accession numbers and CIKs stay in provenance so their digits are never "evidence".
    assert fact.provenance.identifier not in fact.quote


def test_sec_evidence_ids_and_provenance() -> None:
    fact = sec_fact(
        pick(REVENUE_2009, "FY2009", date(2009, 12, 31)), as_of=date(2009, 12, 31), retrieved_at=NOW
    )
    provenance = fact.provenance
    assert provenance.evidence_id == (
        "sec:0000320193:us-gaap:SalesRevenueNet:0001193125-09-214859:2008-09-28_2009-09-26:USD"
    )
    assert provenance.freshness == "superseded"
    assert provenance.superseded_by == "0001193125-10-012091"
    assert provenance.value == Decimal("36537000000") and provenance.unit == "USD"
    assert provenance.published_at == date(2009, 10, 27)
    assert provenance.source_url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000119312509214859/"
        "0001193125-09-214859-index.htm"
    )
    assert "42,905" not in fact.quote  # the later value is never mixed into old evidence
    instant = sec_fact(pick(ASSETS, "2009-09-26", date(2010, 1, 25)), as_of=AS_OF, retrieved_at=NOW)
    assert instant.provenance.evidence_id.endswith(":0001193125-10-012091:2009-09-26:USD")
    assert instant.provenance.amended and instant.provenance.revised_before_as_of


def test_filing_url_points_at_the_index_page() -> None:
    assert filing_url(21344, "0000021344-24-000009") == (
        "https://www.sec.gov/Archives/edgar/data/21344/000002134424000009/"
        "0000021344-24-000009-index.htm"
    )


def test_companyfacts_index_keeps_names_only() -> None:
    index = parse_concept_index(
        parse_json((FIXTURES / "sec" / "companyfacts_0000320193.json").read_bytes())
    )
    assert index["us-gaap"]["InterestExpense"] == "Interest Expense"
    assert "dei" in index


async def test_client_resolves_tickers_ciks_and_names() -> None:
    upstream = FakeUpstream()
    data = external_data(upstream)
    assert data.sec is not None
    assert (await data.sec.resolve("brk.a")).cik == 1067983
    assert (await data.sec.resolve("0000320193")).name == "Apple Inc."
    assert (await data.sec.resolve("Apple")).cik == 320193
    assert (await data.sec.resolve("320193")).tickers == ("AAPL",)
    with pytest.raises(ExternalDataError, match="did you mean Apple Inc."):
        await data.sec.resolve("Appel Inc.")
    with pytest.raises(ExternalDataError, match="No SEC registrant has CIK 0000000042"):
        await data.sec.resolve("42")
    assert upstream.count("company_tickers.json") == 1  # cached
    user_agents = {request.headers["user-agent"] for request in upstream.requests}
    assert user_agents == {"Ledger RAG tusharmahalya@gmail.com"}
    await data.aclose()


async def test_cik_missing_from_the_ticker_map_uses_submissions() -> None:
    upstream = FakeUpstream()
    upstream.overrides["company_tickers.json"] = lambda request: httpx.Response(
        200, json={"0": {"cik_str": 21344, "ticker": "KO", "title": "COCA COLA CO"}}
    )
    data = external_data(upstream)
    assert data.sec is not None
    company = await data.sec.resolve("CIK320193")
    assert (company.name, company.tickers) == ("Apple Inc.", ("AAPL",))
    await data.aclose()


async def test_find_concept_falls_back_to_ifrs_and_suggests_on_miss() -> None:
    upstream = FakeUpstream()
    data = external_data(upstream)
    assert data.sec is not None
    shell = await data.sec.find_concept(1306965, "Revenues")
    assert (shell.taxonomy, shell.concept) == ("ifrs-full", "Revenue")
    with pytest.raises(ExternalDataError) as caught:
        await data.sec.find_concept(320193, "InterestExpence")
    assert caught.value.kind == "not_found"
    assert "try us-gaap:InterestExpense" in caught.value.message
    await data.aclose()


def test_calculator_binds_to_sec_quotes() -> None:
    first = citation_from_fact(sec_fact(pick(INTEREST, "FY2022"), as_of=AS_OF, retrieved_at=NOW), 1)
    second = citation_from_fact(
        sec_fact(pick(INTEREST, "FY2023"), as_of=AS_OF, retrieved_at=NOW), 2
    )
    calculation = run_program(
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
        [first, second],
    )
    assert Decimal(calculation.result).quantize(Decimal("0.0001")) == Decimal("34.1863")
    assert calculation.citation_ids == ["C1", "C2"]
