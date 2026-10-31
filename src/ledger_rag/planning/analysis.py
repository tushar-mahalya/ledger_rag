"""Fast deterministic query analysis: company, period, metric and operation extraction."""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ledger_rag.core.models import Chunk, Document, Principal

Operation = Literal["lookup", "compare", "ratio", "change", "percent_change", "aggregate"]
CALCULATING: frozenset[str] = frozenset({"ratio", "change", "percent_change", "aggregate"})
MULTI_PART: frozenset[str] = frozenset(
    {"compare", "ratio", "change", "percent_change", "aggregate"}
)

_TICKER = re.compile(r"^[A-Z][A-Z0-9.&-]{0,9}$")
# FinQA identities: document id ``ADI/2009/page_49.pdf``; title ``ADI annual report 2009, page 49``.
_DOCUMENT_ID = re.compile(r"^([A-Z][A-Z0-9.&-]{0,9})/((?:19|20)\d\d)/")
_TITLE = re.compile(r"^([A-Z][A-Z0-9.&-]{0,9}) annual report ((?:19|20)\d\d)\b")
_YEAR = re.compile(r"(?<![\d$.,])((?:19[5-9]|20[0-4])\d)(?![\d%])")
_SHORT_FY = re.compile(r"\bfy\s?'?(\d\d)\b", re.I)
_QUARTER = re.compile(r"\b(q[1-4])\b|\b(first|second|third|fourth) quarter\b", re.I)
_UPPER_TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9.&-]{0,9}(?![A-Za-z0-9])")
_FOLLOW_UP = re.compile(
    r"^\s*(?:and|also|what about|how about|same|then|what of)\b"
    r"|\b(?:it|its|it's|they|them|their|those|these|the same|previous year|prior year"
    r"|the year before|the following year|that year|that period|that company)\b",
    re.I,
)
_ORDINALS = {"first": "q1", "second": "q2", "third": "q3", "fourth": "q4"}

# Canonical metric -> surface forms. Longest surface forms are matched first.
METRICS: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue", "revenues", "net sales", "sales", "net revenue", "net revenues"),
    "net income": ("net income", "net earnings", "net profit", "net loss", "earnings"),
    "operating income": ("operating income", "operating profit", "income from operations"),
    "gross margin": ("gross margin", "gross profit"),
    "operating margin": ("operating margin",),
    "eps": ("eps", "earnings per share"),
    "operating cash flow": (
        "cash flow from operations",
        "operating cash flow",
        "cash provided by operating activities",
        "cash from operations",
    ),
    "free cash flow": ("free cash flow",),
    "capital expenditures": ("capital expenditures", "capital expenditure", "capex"),
    "total assets": ("total assets",),
    "debt": ("long-term debt", "total debt", "debt", "borrowings", "notes payable"),
    "dividends": ("dividends", "dividend"),
    "share repurchases": ("share repurchases", "stock repurchases", "buybacks", "repurchased"),
    "research and development": ("research and development", "r&d"),
    "sg&a": ("sg&a", "selling, general and administrative", "general and administrative"),
    "interest expense": ("interest expense",),
    "tax rate": ("effective tax rate", "tax rate"),
    "income taxes": ("income tax expense", "income taxes", "provision for income taxes"),
    "depreciation and amortization": ("depreciation", "amortization"),
    "goodwill": ("goodwill",),
    "intangible assets": ("intangible assets", "intangibles"),
    "inventory": ("inventory", "inventories"),
    "backlog": ("backlog",),
    "ebitda": ("ebitda",),
    "operating expenses": ("operating expenses", "opex"),
    "cash": ("cash and cash equivalents", "cash balance"),
    "stockholders' equity": ("stockholders' equity", "shareholders' equity", "total equity"),
    "rent expense": ("rent expense", "rent expenses", "lease expense"),
    "pension": ("pension contributions", "pension", "postretirement"),
    "stock-based compensation": ("stock-based compensation", "share-based compensation"),
    "restructuring": ("restructuring charges", "restructuring"),
    "accounts receivable": ("accounts receivable", "receivables"),
    "stock return": ("cumulative return", "total return", "stock return", "shareholder return"),
}
_METRIC_PATTERNS = sorted(
    ((form, canonical) for canonical, forms in METRICS.items() for form in forms),
    key=lambda item: -len(item[0]),
)

_OPERATIONS: tuple[tuple[Operation, re.Pattern[str]], ...] = (
    (
        "percent_change",
        re.compile(
            r"\b(?:percent(?:age)?|%)\s+(?:change|increase|decrease|decline|growth)\b"
            r"|\bgrowth rate\b|\bby what percent(?:age)?\b|\bhow much (?:percent|%)"
            r"|\b(?:grow|grew|increase|decrease|decline|change)[a-z]*\s+by\s+what\s+percent",
            re.I,
        ),
    ),
    (
        "ratio",
        re.compile(
            r"\bratio\b|\bas a (?:percent(?:age)?|share|portion|fraction) of\b|\bproportion\b"
            r"|\bportion\b|\bfraction\b|\bpercent(?:age)? of\b|\bshare of\b",
            re.I,
        ),
    ),
    (
        "aggregate",
        re.compile(
            r"\baverage\b|\bmean\b|\bsum of\b|\bcombined\b|\bcumulative (?!return)"
            r"|\bin total\b|\btotal of\b|\baltogether\b|\badd(?:ed)? up\b",
            re.I,
        ),
    ),
    (
        "change",
        re.compile(
            r"\bchange[ds]?\b|\bincrease[ds]?\b|\bdecrease[ds]?\b|\bdifference\b|\bgr[eo]w\b"
            r"|\bgrowth\b|\bdecline[ds]?\b|\bdrop(?:ped)?\b|\brose\b|\bfell\b",
            re.I,
        ),
    ),
    (
        "compare",
        re.compile(
            r"\bcompare[ds]?\b|\bcomparison\b|\bversus\b|\bvs\b\.?|\bhigher\b|\blower\b"
            r"|\bgreater\b|\bmore than\b|\bless than\b|\blarger\b|\bsmaller\b",
            re.I,
        ),
    ),
)


class Gazetteer:
    """Company aliases (lowercase surface form -> ticker) built from corpus metadata."""

    def __init__(self) -> None:
        self.tickers: set[str] = set()
        self.aliases: dict[str, str] = {}

    def add(self, ticker: str, *aliases: str) -> None:
        ticker = ticker.strip().upper()
        if not _TICKER.match(ticker):
            return
        self.tickers.add(ticker)
        for alias in aliases:
            alias = " ".join(alias.lower().split())
            # Short aliases are matched only as uppercase tickers to avoid common words.
            if 4 <= len(alias) <= 100:
                self.aliases[alias] = ticker

    @classmethod
    def from_file(cls, path: Path) -> "Gazetteer":
        """Load ``{"TICKER": ["alias", ...]}`` JSON, e.g. from scripts/build_gazetteer.py."""
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict) or len(raw) > 100_000:
            raise ValueError("Gazetteer must be a bounded JSON object")
        gazetteer = cls()
        for ticker, aliases in raw.items():
            if not isinstance(ticker, str) or not isinstance(aliases, list):
                raise ValueError("Gazetteer entries must map tickers to alias lists")
            gazetteer.add(ticker, *[alias for alias in aliases[:20] if isinstance(alias, str)])
        return gazetteer

    @classmethod
    def from_corpus(cls, items: Iterable[Document | Chunk]) -> "Gazetteer":
        """Tickers from FinQA-style document ids/titles (``ADI/2009/page_49.pdf``)."""
        gazetteer = cls()
        for item in items:
            identifier = item.id if isinstance(item, Document) else item.document_id
            match = _DOCUMENT_ID.match(identifier) or _TITLE.match(item.title)
            if match:
                gazetteer.add(match.group(1))
        return gazetteer

    def with_principal(self, principal: Principal | None, prefix: str) -> "Gazetteer":
        """Add tickers the principal may read (``company:ADI`` groups); never widens access."""
        if principal is None or not prefix:
            return self
        extra = [g[len(prefix) :] for g in principal.groups[:1000] if g.startswith(prefix)]
        if not extra:
            return self
        merged = Gazetteer()
        merged.tickers = set(self.tickers)
        merged.aliases = dict(self.aliases)
        for ticker in extra:
            merged.add(ticker)
        return merged

    def companies(self, text: str) -> list[str]:
        found: list[str] = []
        for token in _UPPER_TOKEN.findall(text):
            token = token.rstrip(".")
            # Single-letter tickers (e.g. "A") only count when written as "A's"-free tokens
            # of length >= 2, so the pronoun "I" or article "A" never matches.
            if len(token) >= 2 and token in self.tickers and token not in found:
                found.append(token)
        if self.aliases:
            lowered = " " + " ".join(re.findall(r"[a-z0-9&.-]+", text.lower())) + " "
            for alias, ticker in self.aliases.items():
                if f" {alias} " in lowered and ticker not in found:
                    found.append(ticker)
        return found


@dataclass(frozen=True)
class QueryAnalysis:
    companies: tuple[str, ...] = ()
    years: tuple[int, ...] = ()
    quarters: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    operation: Operation = "lookup"
    follow_up: bool = False
    confidence: float = 0.0

    @property
    def needs_calculation(self) -> bool:
        return self.operation in CALCULATING

    @property
    def multi_part(self) -> bool:
        return self.operation in MULTI_PART and (len(self.companies) > 1 or len(self.years) > 1)

    @property
    def self_contained(self) -> bool:
        return bool(self.metrics) and bool(self.companies or self.years) and not self.follow_up


def find_years(text: str) -> list[int]:
    found: list[int] = []
    for year in [int(m.group(1)) for m in _YEAR.finditer(text)] + [
        2000 + int(m.group(1)) for m in _SHORT_FY.finditer(text)
    ]:
        if year not in found:
            found.append(year)
    return found


def find_metrics(text: str) -> list[str]:
    consumed = " " + " ".join(re.findall(r"[a-z0-9&'-]+,?", text.lower())) + " "
    consumed = consumed.replace(",", "")
    found: list[str] = []
    for form, canonical in _METRIC_PATTERNS:
        needle = f" {form.replace(',', '')} "
        if needle in consumed:
            consumed = consumed.replace(needle, " # ")
            if canonical not in found:
                found.append(canonical)
    return found


def find_operation(text: str) -> Operation:
    for name, pattern in _OPERATIONS:
        if pattern.search(text):
            return name
    return "lookup"


def analyze(text: str, gazetteer: Gazetteer | None = None) -> QueryAnalysis:
    """Pure, sub-millisecond extraction; never raises on arbitrary user text."""
    companies = (gazetteer or Gazetteer()).companies(text)
    years = find_years(text)
    quarters: list[str] = []
    for match in _QUARTER.finditer(text):
        quarter = (match.group(1) or _ORDINALS[match.group(2).lower()]).lower()
        if quarter not in quarters:
            quarters.append(quarter)
    metrics = find_metrics(text)
    operation = find_operation(text)
    if operation == "lookup" and (len(companies) > 1 or len(years) > 1) and metrics:
        operation = "compare"
    signals = sum(bool(signal) for signal in (companies, years or quarters, metrics))
    follow_up = bool(_FOLLOW_UP.search(text)) or (
        len(text.split()) <= 4 and not (companies and metrics)
    )
    return QueryAnalysis(
        companies=tuple(companies[:8]),
        years=tuple(years[:8]),
        quarters=tuple(quarters[:4]),
        metrics=tuple(metrics[:6]),
        operation=operation,
        follow_up=follow_up,
        confidence=round(min(1.0, 0.3 * signals + (0.1 if operation != "lookup" else 0.0)), 2),
    )
