"""Turn external facts into deterministic citation text plus typed provenance.

The quote is the only text the calculator DSL and numeric verification read, so it is
rendered to contain exactly one reported quantity in a form ``extract_numbers``
understands (``USD 3,933,000,000``, ``5.33%``, ``USD 6.13 per share``, ``USD 27,610.128
billion``). Identifiers whose digits are not quantities (CIKs, accession numbers, units
such as "Index 1982-1984=100") stay in the provenance, never in the quote, and so do
names or titles that would read as numbers ("10-Year", "1-800-FLOWERS.COM").
"""

import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from ledger_rag.core.provenance import Freshness, Provenance
from ledger_rag.external.fred import MacroObservation, series_url
from ledger_rag.external.sec import Selection, filing_url
from ledger_rag.generation.program import extract_numbers

_CURRENCIES = frozenset({"USD", "EUR", "GBP", "JPY"})
_SCALES = {"thousands": (3, "thousand"), "millions": (6, "million"), "billions": (9, "billion")}
_SCALES["trillions"] = (12, "trillion")


@dataclass(frozen=True)
class ExternalFact:
    """One citable external value: a stable ID, a citation title/quote and provenance."""

    title: str
    quote: str
    provenance: Provenance


def number(value: Decimal) -> str:
    """Exact digits with thousands separators; never scientific notation."""
    return format(value, ",f")


def _wordy(text: str | None) -> str | None:
    """``text`` if short and free of numbers the verifier could mistake for evidence."""
    return text if text and len(text) <= 160 and not extract_numbers(text) else None


def _freshness(superseded: bool, stale: bool) -> Freshness:
    return "superseded" if superseded else "stale" if stale else "current"


def sec_value(value: Decimal, unit: str) -> str:
    if unit in _CURRENCIES:
        return f"{unit} {number(value)}"
    if unit == "pure":
        # XBRL "pure" ratios are fractions; the quote states them as percentages.
        return f"{number((value * 100).normalize())}%"
    if unit.endswith("/shares"):
        currency = unit.split("/", 1)[0]
        if currency in _CURRENCIES:
            return f"{currency} {number(value)} per share"
        return f"{number(value)} {_safe_unit(currency)} per share".replace("  ", " ")
    return f"{number(value)} {_safe_unit(unit)}".rstrip()


def _safe_unit(unit: str) -> str:
    return "" if re.search(r"\d", unit) else unit


def sec_fact(selection: Selection, *, as_of: date, retrieved_at: datetime) -> ExternalFact:
    fact, concept = selection.fact, selection.concept
    qualified = f"{concept.taxonomy}:{concept.concept}"
    label = _wordy(concept.label)
    name = f"{label} ({qualified})" if label else qualified
    entity = _wordy(concept.entity) or "The registrant"
    # The quote depends only on the evidence ID's fact, never on how it was requested, so a
    # re-read renders identical text; the requested label (FY2023, CY2023Q4) is provenance.
    when = (
        f"as of {fact.end.isoformat()}"
        if fact.start is None
        else f"for the period {fact.start.isoformat()} to {fact.end.isoformat()}"
    )
    quote = (
        f"{entity} reported {name} of {sec_value(fact.value, fact.unit)} {when}, "
        f"in Form {fact.form} filed {fact.filed.isoformat()}."
    )
    period_key = (
        fact.end.isoformat()
        if fact.start is None
        else f"{fact.start.isoformat()}_{fact.end.isoformat()}"
    )
    superseded = selection.superseded_by
    provenance = Provenance(
        source_kind="sec_xbrl",
        provider="SEC EDGAR",
        evidence_id=(
            f"sec:{concept.cik:010d}:{concept.taxonomy}:{concept.concept}:{fact.accn}:"
            f"{period_key}:{fact.unit}"
        ),
        identifier=fact.accn,
        entity=concept.entity,
        concept=qualified,
        form=fact.form,
        fiscal_year=selection.fiscal_year,
        fiscal_period=selection.fiscal_period,
        period_label=selection.period_label,
        period_start=fact.start,
        period_end=fact.end,
        unit=fact.unit,
        scale=0,
        value=fact.value,
        published_at=fact.filed,
        retrieved_at=retrieved_at,
        as_of=as_of,
        source_url=filing_url(concept.cik, fact.accn),
        freshness=_freshness(superseded is not None, selection.stale),
        amended=fact.amended,
        superseded_by=superseded.accn if superseded else None,
        revised_before_as_of=selection.restated,
    )
    title = f"{concept.entity} {fact.form} filed {fact.filed.isoformat()}: {qualified}"
    return ExternalFact(title=title[:300], quote=quote, provenance=provenance)


def fred_units(units: str) -> tuple[int, str, str]:
    """(scale exponent, scale word, base units) for FRED units such as "Billions of Dollars"."""
    match = re.fullmatch(r"(Thousands|Millions|Billions|Trillions) of (.+)", units.strip(), re.I)
    if not match:
        return 0, "", units.strip()
    exponent, word = _SCALES[match.group(1).lower()]
    return exponent, word, match.group(2).strip()


def fred_value(value: Decimal, units: str) -> str:
    _, word, base = fred_units(units)
    scaled = f"{number(value)} {word}".rstrip() if word else number(value)
    lowered = base.lower()
    if lowered.startswith("percentage point"):
        return f"{number(value)} percentage points"
    if lowered.startswith("percent"):
        return f"{number(value)}%"
    if lowered.startswith("basis point"):
        return f"{number(value)} bps"
    dollars = re.fullmatch(r"(?:u\.s\. )?(?:chained \d{4} )?dollars(?: per (.+))?", lowered)
    if dollars:
        per = dollars.group(1)
        suffix = f" per {per}" if per and not re.search(r"\d", per) else ""
        return f"USD {scaled}{suffix}"
    if lowered.startswith("index"):
        return f"{scaled} (index)"
    return f"{scaled} {_safe_unit(lowered)}".rstrip()


def fred_fact(
    observation: MacroObservation, *, as_of: date, retrieved_at: datetime
) -> ExternalFact:
    series = observation.series
    start, end = series.period_bounds(observation.day)
    exponent, _, _ = fred_units(series.units)
    title = _wordy(series.title)
    named = f"FRED series {series.id} ({title})" if title else f"FRED series {series.id}"
    frequency = f"{cadence} " if (cadence := _wordy(series.frequency.lower())) else ""
    quote = (
        f"{named}, {frequency}observation "
        f"for {observation.day.isoformat()}: {fred_value(observation.value, series.units)}, "
        f"as published in the vintage of {observation.vintage.isoformat()}."
    )
    provenance = Provenance(
        source_kind="fred",
        provider="FRED",
        evidence_id=(
            f"fred:{series.id}:{observation.day.isoformat()}:{observation.vintage.isoformat()}"
        ),
        identifier=series.id,
        entity="Federal Reserve Bank of St. Louis (FRED/ALFRED)",
        concept=series.title,
        period_label=observation.day.isoformat(),
        period_start=start,
        period_end=end,
        unit=series.units,
        scale=exponent,
        value=observation.value,
        published_at=observation.vintage,
        retrieved_at=retrieved_at,
        as_of=as_of,
        source_url=series_url(series.id),
        freshness=_freshness(observation.revised_after is not None, observation.stale),
        superseded_by=observation.revised_after.isoformat() if observation.revised_after else None,
        revised_before_as_of=observation.revised_before,
    )
    return ExternalFact(
        title=f"FRED {series.id}: {series.title}"[:300], quote=quote, provenance=provenance
    )
