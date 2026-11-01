"""SEC EDGAR XBRL facts with point-in-time selection.

Data comes from the official ``data.sec.gov`` APIs (companyconcept, companyfacts,
submissions) and the ``company_tickers.json`` ticker map. In those payloads ``fy``/``fp``
describe the *filing* a fact appeared in, not the fact's own period, and the same period
is re-reported (and sometimes restated) by later filings. Selection therefore works on
the concrete ``(start, end)`` period and only on facts whose ``filed`` date is on or
before the as-of cutoff; later filings are inspected only to flag that a value changed.
"""

import asyncio
import difflib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any, Literal

from ledger_rag.core.provenance import plausible
from ledger_rag.external.errors import ExternalDataError
from ledger_rag.external.http import ExternalHttp, parse_json

SEC_DATA = "https://data.sec.gov"
SEC_WWW = "https://www.sec.gov"
TICKERS_URL = f"{SEC_WWW}/files/company_tickers.json"
TAXONOMIES = ("us-gaap", "ifrs-full", "dei", "srt")
ANNUAL_FORMS = frozenset({"10-K", "10-K/A", "10-KT", "10-KT/A", "20-F", "20-F/A", "40-F", "40-F/A"})
INTERIM_FORMS = frozenset({"10-Q", "10-Q/A", "10-QT", "10-QT/A", "6-K", "6-K/A"})
PERIODIC_FORMS = ANNUAL_FORMS | INTERIM_FORMS
ANNUAL_DAYS = range(335, 381)  # 52/53-week fiscal years are 364 or 371 days long.
QUARTER_DAYS = range(80, 101)  # 13/14-week quarters are 91 or 98 days long.
STALE_ANNUAL_DAYS = 456
STALE_INTERIM_DAYS = 200
# Keeps sec:{cik}:{taxonomy}:{concept}:{accession}:{start}_{end} within 200 characters.
MAX_CONCEPT_LENGTH = 132
CONCEPT_PATTERN = re.compile(r"^(?:(us-gaap|ifrs-full|dei|srt):)?([A-Za-z][A-Za-z0-9_]{0,199})$")
TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9-]{0,9}$")
CIK_PATTERN = re.compile(r"^(?:CIK)?[ \t]*0*([0-9]{1,10})$", re.IGNORECASE)
ACCESSION_PATTERN = re.compile(r"^[0-9]{10}-[0-9]{2}-[0-9]{6}$")
# Conservative equivalents for IFRS filers (20-F/40-F); the actual concept is always shown.
IFRS_ALIASES: Mapping[str, tuple[str, ...]] = {
    "Revenues": ("Revenue",),
    "RevenueFromContractWithCustomerExcludingAssessedTax": ("Revenue",),
    "SalesRevenueNet": ("Revenue",),
    "NetIncomeLoss": ("ProfitLossAttributableToOwnersOfParent", "ProfitLoss"),
    "ProfitLoss": ("ProfitLoss",),
    "OperatingIncomeLoss": ("ProfitLossFromOperatingActivities",),
    "StockholdersEquity": ("EquityAttributableToOwnersOfParent", "Equity"),
    "CashAndCashEquivalentsAtCarryingValue": ("CashAndCashEquivalents",),
    "EarningsPerShareBasic": ("BasicEarningsLossPerShare",),
    "EarningsPerShareDiluted": ("DilutedEarningsLossPerShare",),
    "IncomeTaxExpenseBenefit": ("IncomeTaxExpenseContinuingOperations",),
}
_NAME_SUFFIXES = frozenset(
    {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "plc"}
    | {"llc", "lp", "sa", "ag", "nv", "se", "the"}
)


@dataclass(frozen=True)
class Company:
    cik: int
    name: str
    tickers: tuple[str, ...]

    @property
    def padded(self) -> str:
        return f"{self.cik:010d}"


def parse_cik(text: str) -> int | None:
    match = CIK_PATTERN.fullmatch(text.strip())
    if not match or not int(match.group(1)):
        return None
    return int(match.group(1))


def normalize_ticker(text: str) -> str | None:
    ticker = re.sub(r"[.\s/]+", "-", text.strip().upper())
    return ticker if TICKER_PATTERN.fullmatch(ticker) else None


def normalize_name(text: str) -> str:
    words = re.sub(r"[^a-z0-9]+", " ", text.lower().replace("&", " and ")).split()
    return " ".join(word for word in words if word not in _NAME_SUFFIXES)


class CompanyIndex:
    """Ticker, CIK and exact-name lookup over SEC's ticker map (share classes share a CIK)."""

    def __init__(self, companies: Iterable[Company]) -> None:
        self.by_cik: dict[int, Company] = {}
        self.by_ticker: dict[str, Company] = {}
        self.by_name: dict[str, list[Company]] = {}
        for company in companies:
            self.by_cik[company.cik] = company
            for ticker in company.tickers:
                self.by_ticker.setdefault(ticker, company)
            self.by_name.setdefault(normalize_name(company.name), []).append(company)

    @classmethod
    def from_payload(cls, payload: Any) -> "CompanyIndex":
        if not isinstance(payload, dict):
            raise ValueError("company tickers must be an object")
        names: dict[int, str] = {}
        tickers: dict[int, list[str]] = {}
        for row in payload.values():
            if not isinstance(row, dict):
                continue
            cik, ticker, title = row.get("cik_str"), row.get("ticker"), row.get("title")
            if not isinstance(cik, int) or not isinstance(title, str):
                continue
            if not isinstance(ticker, str) or not 0 < cik < 10**10:
                continue
            names.setdefault(cik, title.strip()[:300])
            normalized = normalize_ticker(ticker)
            if normalized:
                tickers.setdefault(cik, []).append(normalized)
        if not names:
            raise ValueError("company tickers are empty")
        return cls(
            Company(cik=cik, name=name, tickers=tuple(tickers.get(cik, ())))
            for cik, name in names.items()
        )

    def suggestions(self, text: str, limit: int = 5) -> list[Company]:
        names = difflib.get_close_matches(normalize_name(text), self.by_name, n=limit, cutoff=0.75)
        return [self.by_name[name][0] for name in names]


@dataclass(frozen=True)
class Fact:
    unit: str
    value: Decimal
    start: date | None
    end: date
    accn: str
    fy: int | None
    fp: str | None
    form: str
    filed: date
    frame: str | None

    @property
    def days(self) -> int | None:
        return None if self.start is None else (self.end - self.start).days + 1

    @property
    def amended(self) -> bool:
        return self.form.endswith("/A")

    @property
    def period(self) -> tuple[date | None, date]:
        return self.start, self.end


@dataclass(frozen=True)
class ConceptFacts:
    cik: int
    entity: str
    taxonomy: str
    concept: str
    label: str | None
    facts: tuple[Fact, ...]


def _number(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, int | Decimal):
        return None
    number = Decimal(value)
    return number if plausible(number) else None


def _date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def parse_concept(payload: Any) -> ConceptFacts:
    """Validate a companyconcept document; malformed individual facts are skipped."""
    if not isinstance(payload, dict) or not isinstance(payload.get("units"), dict):
        raise ValueError("companyconcept payload has no units")
    cik, taxonomy, tag = payload.get("cik"), payload.get("taxonomy"), payload.get("tag")
    entity, label = payload.get("entityName"), payload.get("label")
    if not isinstance(cik, int) or taxonomy not in TAXONOMIES or not isinstance(tag, str):
        raise ValueError("companyconcept payload has no identity")
    facts: list[Fact] = []
    for unit, rows in payload["units"].items():
        if not isinstance(unit, str) or not isinstance(rows, list) or len(unit) > 60:
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            value, end, filed = (
                _number(row.get("val")),
                _date(row.get("end")),
                _date(row.get("filed")),
            )
            start = _date(row.get("start")) if "start" in row else None
            accn, form = row.get("accn"), row.get("form")
            if value is None or end is None or filed is None:
                continue
            if not isinstance(accn, str) or not ACCESSION_PATTERN.fullmatch(accn):
                continue
            if not isinstance(form, str) or len(form) > 16 or ("start" in row and start is None):
                continue
            if start is not None and start > end:
                continue
            fy, fp, frame = row.get("fy"), row.get("fp"), row.get("frame")
            facts.append(
                Fact(
                    unit=unit,
                    value=value,
                    start=start,
                    end=end,
                    accn=accn,
                    fy=fy if isinstance(fy, int) and not isinstance(fy, bool) else None,
                    fp=fp if isinstance(fp, str) and len(fp) <= 4 else None,
                    form=form,
                    filed=filed,
                    frame=frame if isinstance(frame, str) and len(frame) <= 12 else None,
                )
            )
    return ConceptFacts(
        cik=cik,
        entity=entity.strip()[:300] if isinstance(entity, str) and entity.strip() else f"CIK {cik}",
        taxonomy=taxonomy,
        concept=tag,
        label=label.strip()[:300] if isinstance(label, str) and label.strip() else None,
        facts=tuple(facts),
    )


def parse_concept_index(payload: Any) -> dict[str, dict[str, str]]:
    """Concept names and labels per taxonomy from companyfacts (values are discarded)."""
    if not isinstance(payload, dict) or not isinstance(payload.get("facts"), dict):
        raise ValueError("companyfacts payload has no facts")
    index: dict[str, dict[str, str]] = {}
    for taxonomy, concepts in payload["facts"].items():
        if taxonomy not in TAXONOMIES or not isinstance(concepts, dict):
            continue
        labels: dict[str, str] = {}
        for name, body in concepts.items():
            if (
                not isinstance(name, str)
                or not CONCEPT_PATTERN.fullmatch(name)
                or len(name) > MAX_CONCEPT_LENGTH
            ):
                continue
            label = body.get("label") if isinstance(body, dict) else None
            labels[name] = label[:300] if isinstance(label, str) and label else name
        index[taxonomy] = labels
    return index


PeriodKind = Literal["fy", "fq", "cy", "date", "latest"]


@dataclass(frozen=True)
class PeriodSpec:
    kind: PeriodKind
    year: int = 0
    quarter: int = 0
    instant: bool = False
    day: date | None = None

    @property
    def label(self) -> str:
        if self.kind == "fy":
            return f"FY{self.year}"
        if self.kind == "fq":
            return f"FY{self.year}Q{self.quarter}"
        if self.kind == "cy":
            quarter = f"Q{self.quarter}" if self.quarter else ""
            return f"CY{self.year}{quarter}{'I' if self.instant else ''}"
        if self.kind == "date" and self.day is not None:
            return self.day.isoformat()
        return "latest"


PERIOD_HELP = (
    "periods must be FY2023, FY2023Q2, CY2023, CY2023Q4, CY2023Q4I, YYYY-MM-DD "
    "(balance-sheet dates) or latest"
)


def parse_period(text: str) -> PeriodSpec:
    raw = text.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        day = _date(raw)
        if day is None:
            raise ExternalDataError("invalid_request", PERIOD_HELP)
        return PeriodSpec("date", day=day)
    compact = re.sub(r"[\s_-]+", "", raw.upper())
    match = re.fullmatch(r"(?:FY(\d{4})(?:Q([1-4]))?|Q([1-4])FY(\d{4}))", compact)
    if match:
        year = int(match.group(1) or match.group(4))
        quarter = int(match.group(2) or match.group(3) or 0)
        spec = PeriodSpec("fq" if quarter else "fy", year=year, quarter=quarter)
    elif match := re.fullmatch(r"CY(\d{4})(?:Q([1-4]))?(I)?", compact):
        if match.group(3) and not match.group(2):
            raise ExternalDataError("invalid_request", PERIOD_HELP)
        spec = PeriodSpec(
            "cy",
            year=int(match.group(1)),
            quarter=int(match.group(2) or 0),
            instant=bool(match.group(3)),
        )
    elif compact == "LATEST":
        return PeriodSpec("latest")
    else:
        raise ExternalDataError("invalid_request", PERIOD_HELP)
    if not 1990 <= spec.year <= 2100:
        raise ExternalDataError("invalid_request", "period years must be between 1990 and 2100")
    return spec


class SelectionMiss(Exception):
    """No point-in-time value; the message is safe to show the model and user."""


@dataclass(frozen=True)
class Selection:
    fact: Fact
    concept: ConceptFacts
    spec: PeriodSpec
    period_label: str
    fiscal_year: int | None
    fiscal_period: str | None
    restated: bool
    superseded_by: Fact | None
    stale: bool


def unit_order(units: Iterable[str], requested: str | None) -> list[str]:
    available = sorted(set(units))
    if requested:
        wanted = requested.strip().replace("-per-", "/").lower()
        return [unit for unit in available if unit.lower() == wanted]

    def rank(unit: str) -> tuple[int, str]:
        if unit == "USD":
            return 0, unit
        if re.fullmatch(r"[A-Z]{3}", unit):
            return 1, unit
        if unit.endswith("/shares"):
            return 2, unit
        if unit == "shares":
            return 3, unit
        if unit == "pure":
            return 4, unit
        return 5, unit

    return sorted(available, key=rank)


def _annual(fact: Fact) -> bool:
    return fact.days is not None and fact.days in ANNUAL_DAYS


def _quarterly(fact: Fact) -> bool:
    return fact.days is not None and fact.days in QUARTER_DAYS


def _latest(facts: Iterable[Fact]) -> Fact:
    # Same-day ties prefer the amendment, then the later accession number.
    return max(facts, key=lambda fact: (fact.filed, fact.amended, fact.accn))


def select_fact(
    concept: ConceptFacts, spec: PeriodSpec, as_of: date, unit: str | None = None
) -> Selection:
    """Pick the value known at ``as_of`` for ``spec``, trying units in preference order."""
    units = unit_order((fact.unit for fact in concept.facts), unit)
    if not units:
        raise SelectionMiss(
            f"{concept.taxonomy}:{concept.concept} has no values in unit {unit}"
            if unit
            else f"{concept.taxonomy}:{concept.concept} has no values"
        )
    misses: list[str] = []
    for candidate in units:
        facts = [
            fact for fact in concept.facts if fact.unit == candidate and fact.form in PERIODIC_FORMS
        ]
        try:
            return _select(concept, facts, spec, as_of)
        except SelectionMiss as miss:
            misses.append(str(miss))
    raise SelectionMiss(misses[0] if misses else "no periodic-report values")


def _select(concept: ConceptFacts, facts: list[Fact], spec: PeriodSpec, as_of: date) -> Selection:
    if not facts:
        raise SelectionMiss("no values from periodic reports (10-K, 10-Q, 20-F, 40-F, 6-K)")
    instant = all(fact.start is None for fact in facts)
    if not instant:
        facts = [fact for fact in facts if fact.start is not None]
    known = [fact for fact in facts if fact.filed <= as_of]
    name = f"{concept.taxonomy}:{concept.concept}"
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    label = spec.label
    stale = False
    if spec.kind in {"fy", "fq"}:
        fiscal_year = spec.year
        fiscal_period = f"Q{spec.quarter}" if spec.kind == "fq" else "FY"
        period = _fiscal_period(known, spec, instant, as_of)
    elif spec.kind == "cy":
        if spec.instant and not instant:
            raise SelectionMiss(f"{name} is a duration concept; drop the trailing I")
        frame = spec.label
        if instant and not spec.instant:
            frame = f"CY{spec.year}Q{spec.quarter or 4}I"
        label = frame
        framed = [fact for fact in facts if fact.frame == frame]
        if not framed:
            raise SelectionMiss(f"no {name} value is aligned to calendar frame {frame}")
        period = framed[0].period
    elif spec.kind == "date":
        if not instant:
            raise SelectionMiss(
                f"{name} measures a period; dates select point-in-time balances, so use a "
                "fiscal (FY2023, FY2023Q2) or calendar (CY2023) period"
            )
        assert spec.day is not None
        period = (None, spec.day)
    else:
        pool = known if instant else [f for f in known if f.form in ANNUAL_FORMS and _annual(f)]
        if not pool:
            raise SelectionMiss(f"no {name} value had been filed as of {as_of.isoformat()}")
        end = max(fact.end for fact in pool)
        period = _latest(fact for fact in pool if fact.end == end).period
        label = _interval(period)
        threshold = STALE_ANNUAL_DAYS if not instant else STALE_INTERIM_DAYS
        stale = (as_of - end).days > threshold
    candidates = [fact for fact in known if fact.period == period]
    if not candidates:
        raise SelectionMiss(f"no {name} value for {label} had been filed as of {as_of.isoformat()}")
    chosen = _latest(candidates)
    if len({fact.value for fact in candidates if fact.accn == chosen.accn}) > 1:
        raise SelectionMiss(
            f"filing {chosen.accn} reports several {name} values for {label} (for example "
            "one per share class); they cannot be attributed without dimensions"
        )
    later = sorted(
        (fact for fact in facts if fact.period == period and fact.filed > as_of),
        key=lambda fact: (fact.filed, fact.accn),
    )
    return Selection(
        fact=chosen,
        concept=concept,
        spec=spec,
        period_label=label,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        restated=any(fact.value != chosen.value for fact in candidates),
        superseded_by=next((fact for fact in later if fact.value != chosen.value), None),
        stale=stale,
    )


def _interval(period: tuple[date | None, date]) -> str:
    start, end = period
    return end.isoformat() if start is None else f"{start.isoformat()}/{end.isoformat()}"


def _fiscal_period(
    known: list[Fact], spec: PeriodSpec, instant: bool, as_of: date
) -> tuple[date | None, date]:
    """Resolve FY/FQ to concrete dates using the filing(s) that report that fiscal period."""
    cutoff = as_of.isoformat()
    if spec.kind == "fy" or spec.quarter == 4:
        annual = [
            fact
            for fact in known
            if fact.fy == spec.year
            and fact.fp == "FY"
            and fact.form in ANNUAL_FORMS
            and (instant or _annual(fact))
        ]
        if not annual:
            raise SelectionMiss(f"no FY{spec.year} annual report had been filed as of {cutoff}")
        end = max(fact.end for fact in annual)
        if instant:
            return None, end
        if spec.kind == "fy":
            return _latest(fact for fact in annual if fact.end == end).period
        quarters = [fact for fact in known if fact.end == end and _quarterly(fact)]
        if not quarters:
            raise SelectionMiss(
                f"no discrete fiscal Q4 value is filed for FY{spec.year}; derive it from FY"
                f"{spec.year} and the FY{spec.year}Q1-Q3 values with the calculator"
            )
        return _latest(quarters).period
    interim = [
        fact
        for fact in known
        if fact.fy == spec.year
        and fact.fp == f"Q{spec.quarter}"
        and fact.form in INTERIM_FORMS
        and (instant or _quarterly(fact))
    ]
    if not interim:
        raise SelectionMiss(
            f"no FY{spec.year}Q{spec.quarter} quarterly report had been filed as of {cutoff}"
        )
    end = max(fact.end for fact in interim)
    return _latest(fact for fact in interim if fact.end == end).period


def concept_candidates(text: str) -> list[tuple[str, str]]:
    """(taxonomy, concept) pairs to try: explicit taxonomy, else US-GAAP then IFRS then DEI."""
    match = CONCEPT_PATTERN.fullmatch(text.strip())
    if not match:
        raise ExternalDataError(
            "invalid_request",
            "concepts must be XBRL element names such as InterestExpense or us-gaap:Revenues",
        )
    taxonomy, name = match.group(1), match.group(2)
    if len(name) > MAX_CONCEPT_LENGTH:
        raise ExternalDataError(
            "invalid_request", f"concept names are at most {MAX_CONCEPT_LENGTH} characters"
        )
    if taxonomy:
        return [(taxonomy, name)]
    pairs = [("us-gaap", name), ("ifrs-full", name)]
    pairs += [("ifrs-full", alias) for alias in IFRS_ALIASES.get(name, ()) if alias != name]
    pairs.append(("dei", name))
    return list(dict.fromkeys(pairs))


class SecClient:
    """Read-only SEC EDGAR access; all URLs are fixed official endpoints."""

    def __init__(self, http: ExternalHttp) -> None:
        self.http = http

    async def companies(self) -> CompanyIndex:
        return await self.http.get(
            TICKERS_URL,
            key="sec:tickers",
            endpoint="company_tickers",
            parse=lambda raw: CompanyIndex.from_payload(parse_json(raw)),
        )

    async def submissions_company(self, cik: int) -> Company:
        def parse(raw: bytes) -> Company:
            payload = parse_json(raw)
            if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
                raise ValueError("submissions payload has no name")
            tickers = payload.get("tickers")
            return Company(
                cik=cik,
                name=payload["name"].strip()[:300] or f"CIK {cik}",
                tickers=tuple(
                    ticker
                    for ticker in (
                        normalize_ticker(item) for item in tickers or [] if isinstance(item, str)
                    )
                    if ticker
                )
                if isinstance(tickers, list)
                else (),
            )

        return await self.http.get(
            f"{SEC_DATA}/submissions/CIK{cik:010d}.json",
            key=f"sec:submissions:{cik}",
            endpoint="submissions",
            parse=parse,
        )

    async def resolve(self, text: str) -> Company:
        query = text.strip()
        if not query or len(query) > 200:
            raise ExternalDataError("invalid_request", "company must be a ticker, CIK or name")
        index = await self.companies()
        cik = parse_cik(query)
        if cik is not None:
            company = index.by_cik.get(cik)
            if company is not None:
                return company
            try:
                return await self.submissions_company(cik)
            except ExternalDataError as error:
                if error.kind == "not_found":
                    raise ExternalDataError(
                        "not_found", f"No SEC registrant has CIK {cik:010d}"
                    ) from None
                raise
        ticker = normalize_ticker(query)
        if ticker and ticker in index.by_ticker:
            return index.by_ticker[ticker]
        matches = index.by_name.get(normalize_name(query), [])
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise ExternalDataError(
                "invalid_request",
                "Several SEC registrants match that name; use a ticker or CIK: "
                + _describe(matches[:5]),
            )
        suggestions = index.suggestions(query)
        hint = f"; did you mean {_describe(suggestions)}" if suggestions else ""
        raise ExternalDataError("not_found", f"Unknown company{hint}")

    async def concept(self, cik: int, taxonomy: str, concept: str) -> ConceptFacts:
        return await self.http.get(
            f"{SEC_DATA}/api/xbrl/companyconcept/CIK{cik:010d}/{taxonomy}/{concept}.json",
            key=f"sec:concept:{cik}:{taxonomy}:{concept}",
            endpoint="companyconcept",
            parse=lambda raw: parse_concept(parse_json(raw)),
        )

    async def concept_index(self, cik: int) -> dict[str, dict[str, str]]:
        return await self.http.get(
            f"{SEC_DATA}/api/xbrl/companyfacts/CIK{cik:010d}.json",
            key=f"sec:companyfacts:{cik}",
            endpoint="companyfacts",
            parse=lambda raw: parse_concept_index(parse_json(raw)),
        )

    async def frame(
        self, taxonomy: str, concept: str, unit: str, frame: str
    ) -> list[dict[str, Any]]:
        """Cross-company frame snapshot. Not offered as a tool: rows carry no ``filed``
        date per company version, so a frame cannot be cut off at an as-of date."""
        if not CONCEPT_PATTERN.fullmatch(concept) or taxonomy not in TAXONOMIES:
            raise ExternalDataError("invalid_request", "invalid frame concept")
        if not re.fullmatch(r"CY\d{4}(?:Q[1-4]I?)?", frame) or not re.fullmatch(
            r"[A-Za-z]{1,12}(?:/[A-Za-z]{1,12})?", unit
        ):
            raise ExternalDataError("invalid_request", "invalid frame or unit")

        def parse(raw: bytes) -> list[dict[str, Any]]:
            payload = parse_json(raw)
            rows = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                raise ValueError("frames payload has no data")
            return [row for row in rows if isinstance(row, dict)]

        return await self.http.get(
            f"{SEC_DATA}/api/xbrl/frames/{taxonomy}/{concept}/{unit}/{frame}.json",
            key=f"sec:frame:{taxonomy}:{concept}:{unit}:{frame}",
            endpoint="frames",
            parse=parse,
        )

    async def find_concept(self, cik: int, text: str) -> ConceptFacts:
        """First taxonomy that has the concept; a miss lists close concept names."""
        for taxonomy, name in concept_candidates(text):
            try:
                return await self.concept(cik, taxonomy, name)
            except ExternalDataError as error:
                if error.kind != "not_found":
                    raise
        hint = await self._suggest(cik, text)
        raise ExternalDataError(
            "not_found",
            f"Company has no XBRL facts for {text.strip()}" + (f"; try {hint}" if hint else ""),
        )

    async def _suggest(self, cik: int, text: str) -> str:
        try:
            index = await self.concept_index(cik)
        except ExternalDataError:
            return ""
        name = text.strip().split(":")[-1]
        names = {
            f"{taxonomy}:{concept}": concept
            for taxonomy, concepts in index.items()
            for concept in concepts
        }
        close = await asyncio.to_thread(
            difflib.get_close_matches, name, list(names.values()), 5, 0.6
        )
        qualified = [key for want in close for key, value in names.items() if value == want]
        return ", ".join(dict.fromkeys(qualified[:5]))


def _describe(companies: list[Company]) -> str:
    return ", ".join(
        f"{company.name} ({'/'.join(company.tickers[:3]) or 'no ticker'}, CIK {company.padded})"
        for company in companies
    )


def filing_url(cik: int, accession: str) -> str:
    return f"{SEC_WWW}/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{accession}-index.htm"
