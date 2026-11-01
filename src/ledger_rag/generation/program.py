"""Evidence-bound arithmetic programs (FinQA-style DSL) and financial number normalization.

Every operand is bound to evidence: an exact quoted span of a cited passage, a prior
step result (``#0`` is the first step, as in FinQA) or a small whitelisted constant.
Free numeric literals are rejected so the model cannot introduce unsupported inputs.
Arithmetic uses ``Decimal`` with 28 significant digits and ROUND_HALF_EVEN; results are
displayed with at most 10 decimal places.

Every step result also carries a deterministic unit derived from its operands: the
currency, per-share, count or unitless dimension, an ISO currency, a display scale and a
display hint. Units are never converted: amounts in different currencies fail closed.
A bare ``$`` is read as USD (the corpus is SEC filings); ``C$``, ``HK$`` or ``\u00a5``
stay ambiguous and never match an ISO currency.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from typing import Any, Literal

from ledger_rag.core.models import (
    Calculation,
    CalculationKind,
    CalculationOperand,
    CalculationStep,
    CalculationUnit,
    Citation,
    UnitDisplay,
    UnitKind,
    UnitScale,
)

MAX_STEPS = 8
MAX_ARGS = 16
MAX_TOTAL_ARGS = 48
MAX_QUOTE_CHARS = 200
MAX_EXPONENT = 10
MAX_MAGNITUDE = Decimal("1e30")
MIN_NONZERO = Decimal("1e-20")
DISPLAY_PLACES = Decimal("1e-10")
CONSTANTS = frozenset(Decimal(value) for value in ("1", "2", "4", "12", "100", "1000", "1000000"))
BINARY_OPS = frozenset(
    {"add", "subtract", "multiply", "divide", "exp", "greater", "percent_change"}
)
TABLE_OPS = frozenset({"table_sum", "table_average", "table_max", "table_min"})
OPS = tuple(sorted(BINARY_OPS | TABLE_OPS))

Unit = Literal["percent", "percentage_points", "bps", "currency"]

_SCALES = {
    "thousand": Decimal(1000),
    "k": Decimal(1000),
    "million": Decimal(1_000_000),
    "mn": Decimal(1_000_000),
    "m": Decimal(1_000_000),
    "billion": Decimal(1_000_000_000),
    "bn": Decimal(1_000_000_000),
    "b": Decimal(1_000_000_000),
    "trillion": Decimal(1_000_000_000_000),
}
_SCALE_NAMES = {
    "k": "thousand",
    "mn": "million",
    "m": "million",
    "bn": "billion",
    "b": "billion",
}
_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
# Spans whose digits are identifiers, dates or layout rather than reported quantities.
_MASKS = re.compile(
    r"\[C\d+\]"
    r"|^[ \t]*(?:\d{1,2}[.)]|[-*\u2022])(?=[ \t])"
    r"|\b(?:10-K|10-Q|8-K|20-F|40-F|11-K|S-[1-4]|S-K|S-X|10-K405)\b"
    r"|\b(?:Item|Items|Note|Notes|Section|Sections|Part|Rule|Form|Schedule|Exhibit|Article|"
    r"Topic|ASC|ASU|SFAS|FAS|IAS|IFRS|Level|Tier|Page|Footnote|Chapter)\.?[ \t]+"
    r"(?:No\.[ \t]*)?\d+[A-Za-z]?(?:[-.]\d+[A-Za-z]?)*\b"
    r"|\b" + _MONTH + r"\.?[ \t]+\d{1,2}(?:st|nd|rd|th)?\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?[ \t]+" + _MONTH + r"\b"
    r"|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|\b\d+(?:st|nd|rd|th)\b"
    r"|\b(?:FY|CY|Q|H)\d+\b",
    re.IGNORECASE | re.MULTILINE,
)
_NUMBER = re.compile(
    r"(?P<lead>US\$|\$)?"
    r"(?P<open>\([ \t]*)?"
    r"(?:(?<![\w.,\-\u2212])(?P<sign>[-\u2212])[ \t]?)?"
    r"(?:(?P<currency>US\$|\$|\u20ac|\u00a3|\u00a5|(?:USD|EUR|GBP|JPY)[ \t]?)"
    r"(?P<sign2>[-\u2212])?)?"
    r"(?<![\w.])(?<![\d,])"
    r"(?P<core>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?![\d]|[.,]\d)"
    r"(?P<unit>[ \t]?(?:%|percent\b|per[ \t]cent\b|percentage[ \t]points?\b|pp\b|bps\b|"
    r"basis[ \t]points?\b))?"
    r"(?P<scale>[ \t]?(?:thousand|million|billion|trillion|mn|bn)\b|(?:k|m|b)\b)?"
    r"(?P<close>[ \t]*\))?"
    r"(?P<scale2>[ \t](?:thousand|million|billion|trillion)\b)?"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_YEAR = re.compile(r"(?:19|20)\d{2}")
# Loose (answer-side) mode: numbers glued to letters such as "3.5x", "12k" or "40m".
_GLUED = re.compile(
    r"(?<![\w.,])(?P<core>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?P<suffix>[A-Za-z]+)"
)
_GLUED_SCALES = {"k": "k", "m": "m", "mm": "m", "mn": "mn", "b": "b", "bn": "bn"}
_SYMBOLS = {"US$": "USD", "\u20ac": "EUR", "\u00a3": "GBP"}
_LETTERS_BEFORE = re.compile(r"[A-Za-z]+$")


class ProgramError(ValueError):
    """A model-correctable program error; messages never echo raw model input."""


@dataclass(frozen=True)
class NumberMention:
    start: int
    end: int
    core_start: int
    core_end: int
    text: str
    value: Decimal
    scale: Decimal
    scale_word: str | None
    unit: Unit | None
    decimals: int
    year_like: bool
    # Raw currency token: an ISO code, "$" for a bare dollar, or an ambiguous "C$"/"\u00a5".
    currency: str | None = None

    @property
    def absolute(self) -> Decimal:
        return self.value * self.scale

    @property
    def operand_value(self) -> Decimal:
        """Arithmetic value: scaled amounts in base units and percentages as fractions."""
        if self.unit in {"percent", "percentage_points"}:
            return self.value / 100
        if self.unit == "bps":
            return self.value / 10000
        return self.absolute


def _mask(text: str) -> str:
    return _MASKS.sub(lambda match: " " * len(match.group(0)), text)


def extract_numbers(
    text: str, *, include_years: bool = True, loose: bool = False
) -> list[NumberMention]:
    """Normalize financial numbers in text; offsets refer to the original string.

    ``loose`` (used by answer verification) also returns numbers glued to letters
    ("3.5x", "12k", "40m") so they are checked instead of silently skipped; the
    program DSL uses strict mode so such tokens can never be quoted as operands.

    Parentheses mark a negative only around a bare amount, e.g. ``(1,234)`` or
    ``$(1,234)``; prose parentheticals with scale words or percentages stay positive.
    """
    mentions: list[NumberMention] = []
    masked = _mask(text)
    for match in _NUMBER.finditer(masked):
        core = match.group("core")
        unit_text = re.sub(r"\s+", " ", (match.group("unit") or "").strip().lower())
        bracketed = bool(match.group("open") and match.group("close"))
        inner_scale = (match.group("scale") or "").strip().lower()
        outer_scale = (match.group("scale2") or "").strip().lower() if bracketed else ""
        scale_key = inner_scale or outer_scale
        symbol_group = "lead" if match.group("lead") else "currency"
        symbol = (match.group(symbol_group) or "").strip().upper()
        currency = bool(symbol)
        if scale_key in {"k", "m", "b"} and not currency and not loose:
            continue  # "10b-5" or "3m" without a currency is an identifier, not an amount.
        signed = bool(match.group("sign") or match.group("sign2"))
        year_like = bool(_YEAR.fullmatch(core)) and not (
            currency or unit_text or scale_key or signed
        )
        if year_like and not include_years:
            continue
        accounting_negative = bracketed and not (unit_text or inner_scale or year_like)
        unit: Unit | None
        if unit_text in {"%", "percent", "per cent"}:
            unit = "percent"
        elif unit_text.startswith("percentage") or unit_text == "pp":
            unit = "percentage_points"
        elif unit_text:
            unit = "bps"
        else:
            unit = "currency" if currency else None
        value = Decimal(core.replace(",", ""))
        if accounting_negative or signed:
            value = -value
        start, end = match.start(), match.end()
        if match.group("scale2") and not bracketed:
            end = match.start("scale2")
        if not bracketed:
            if match.group("open"):
                start = match.end("open")
            if match.group("close"):
                end = match.start("close")
        mentions.append(
            NumberMention(
                start=start,
                end=end,
                core_start=match.start("core"),
                core_end=match.end("core"),
                text=text[start:end].strip(),
                value=value,
                scale=_SCALES.get(scale_key, Decimal(1)),
                scale_word=_SCALE_NAMES.get(scale_key, scale_key or None),
                unit=unit,
                decimals=len(core.split(".", 1)[1]) if "." in core else 0,
                year_like=year_like,
                currency=_currency_token(masked, match.start(symbol_group), symbol)
                if symbol
                else None,
            )
        )
    if loose:
        mentions.extend(_glued_numbers(text, masked, mentions, include_years))
        mentions.sort(key=lambda mention: mention.start)
    return mentions


def _currency_token(masked: str, start: int, symbol: str) -> str:
    """Letters glued to a dollar or yen sign (C$, HK$, A$) make it ambiguous, never USD."""
    prefix = _LETTERS_BEFORE.search(masked[max(0, start - 8) : start])
    if prefix and symbol in {"$", "\u00a5"}:
        return prefix.group(0)[-3:].upper() + symbol
    return _SYMBOLS.get(symbol, symbol)


def _glued_numbers(
    text: str, masked: str, found: Sequence[NumberMention], include_years: bool
) -> list[NumberMention]:
    taken = [(mention.core_start, mention.core_end) for mention in found]
    glued: list[NumberMention] = []
    for match in _GLUED.finditer(masked):
        start, end = match.span("core")
        if any(start < taken_end and taken_start < end for taken_start, taken_end in taken):
            continue
        core = match.group("core")
        year_like = bool(_YEAR.fullmatch(core))  # "1990s", "2009E"
        if year_like and not include_years:
            continue
        scale_key = _GLUED_SCALES.get(match.group("suffix").lower(), "")
        glued.append(
            NumberMention(
                start=start,
                end=match.end(),
                core_start=start,
                core_end=end,
                text=text[start : match.end()],
                value=Decimal(core.replace(",", "")),
                scale=_SCALES.get(scale_key, Decimal(1)),
                scale_word=_SCALE_NAMES.get(scale_key, scale_key or None),
                unit=None,
                decimals=len(core.split(".", 1)[1]) if "." in core else 0,
                year_like=year_like,
            )
        )
    return glued


_Kind = Literal["currency", "amount", "ratio", "percent", "boolean", "constant"]
_Dim = Literal[
    "currency", "per_share", "count", "unitless", "ratio", "percent", "boolean", "constant"
]
_KINDS: dict[_Dim, _Kind] = {
    "currency": "currency",
    "per_share": "currency",
    "count": "amount",
    "unitless": "amount",
    "ratio": "ratio",
    "percent": "percent",
    "boolean": "boolean",
    "constant": "constant",
}
_MONEY = frozenset({"currency", "per_share"})
_UNIT_SCALES: dict[str, UnitScale] = {
    "thousand": "thousand",
    "million": "million",
    "billion": "billion",
    "trillion": "trillion",
}
_ISO_CURRENCY = re.compile(r"[A-Z]{3}")
_CURRENCY_CODES = frozenset(
    "USD EUR GBP JPY CHF CAD AUD NZD CNY HKD SGD INR KRW SEK NOK DKK BRL MXN ZAR".split()
)
_CODE_BEFORE = re.compile(r"(?<![A-Za-z])([A-Z]{3})[ \t]?$")
_CURRENCY_AFTER = re.compile(
    r"[ \t]*(?:(?P<code>[A-Z]{3})\b|(?:(?P<qualifier>[A-Za-z.]+)[ \t]+)?"
    r"(?P<word>(?i:dollars|euros?|yen))\b)",
)
_US = frozenset({"U.S.", "US", "UNITED"})
_PER_SHARE = re.compile(
    r"[ \t]*(?:(?:per|a)[ \t]+|/[ \t]*)(?:(?:basic|diluted|common|ordinary|and)[ \t]+)*share\b",
    re.IGNORECASE,
)
_COUNT = re.compile(
    r"[ \t]*(?:(?:basic|diluted|weighted[- ]average|common|ordinary|outstanding|full[- ]time|"
    r"net|new|active|total)[ \t]+)*(?P<noun>shares|employees|customers|subscribers|users|"
    r"members|stores|locations|units|vehicles|accounts|restaurants|branches|clients)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Unit:
    dim: _Dim
    # Raw token: an ISO code or an ambiguous symbol such as "C$"; never converted.
    currency: str | None = None
    scale: UnitScale | None = None
    shares: bool = False
    display: UnitDisplay = "amount"

    @property
    def kind(self) -> _Kind:
        if self.dim in _MONEY and self.currency is None:
            return "amount"
        return _KINDS[self.dim]


_CONSTANT = _Unit("constant")
_BOOLEAN = _Unit("boolean", display="boolean")
_PERCENT = _Unit("percent", display="percent")


@dataclass(frozen=True)
class _Value:
    value: Decimal
    operand: CalculationOperand
    citation_ids: frozenset[str]
    unit: _Unit
    evidence_backed: bool

    @property
    def kind(self) -> _Kind:
        return self.unit.kind


def _nearby_currency(text: str, mention: NumberMention) -> str | None:
    """An ISO code or currency word written beside a number ("CAD 5", "5 million euros")."""
    before = _CODE_BEFORE.search(text[max(0, mention.start - 5) : mention.start])
    if before and before.group(1) in _CURRENCY_CODES:
        return before.group(1)
    after = _CURRENCY_AFTER.match(text, mention.end)
    if after is None:
        return None
    if after.group("code"):
        return after.group("code") if after.group("code") in _CURRENCY_CODES else None
    word = after.group("word").lower()
    qualifier = (after.group("qualifier") or "").upper()
    if word == "dollars" and qualifier and qualifier not in _US:
        return f"{qualifier[:8]} $"  # Canadian dollars and the like stay ambiguous.
    return {"dollars": "USD", "yen": "JPY"}.get(word, "EUR")


def _evidence_unit(text: str, mention: NumberMention) -> _Unit:
    if mention.unit is not None and mention.unit != "currency":
        return _Unit("ratio", display="percent")  # Percent-family evidence is a fraction.
    scale = _UNIT_SCALES.get(mention.scale_word or "")
    nearby = _nearby_currency(text, mention)
    currency = mention.currency
    if currency is None or (currency == "$" and nearby is not None):
        currency = nearby
    elif currency == "$":
        currency = "USD"
    tail = text[mention.end : mention.end + 80]
    if _PER_SHARE.match(tail):
        return _Unit("per_share", currency=currency, scale=scale)
    if currency is not None:
        return _Unit("currency", currency=currency, scale=scale)
    count = _COUNT.match(tail)
    if count:
        return _Unit("count", scale=scale, shares=count.group("noun").lower() == "shares")
    return _Unit("unitless", scale=scale)


def _context() -> Context:
    return Context(prec=28, rounding=ROUND_HALF_EVEN, Emax=999, Emin=-999)


def _bounded(value: Decimal, step: int) -> Decimal:
    if not value.is_finite() or abs(value) > MAX_MAGNITUDE:
        raise ProgramError(f"Step #{step} result is outside the supported magnitude (1e30)")
    if value and abs(value) < MIN_NONZERO:
        raise ProgramError(f"Step #{step} result is too small to represent reliably")
    return value


def format_decimal(value: Decimal) -> str:
    with localcontext(_context()):
        rounded = value.quantize(DISPLAY_PLACES) if abs(value) < Decimal("1e17") else value
        text = format(rounded, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def _bind_evidence(raw: Mapping[str, Any], step: int, citations: Mapping[str, Citation]) -> _Value:
    citation_id, quote = raw.get("citation_id"), raw.get("quote")
    if not isinstance(citation_id, str) or citation_id not in citations:
        raise ProgramError(f"Step #{step}: citation_id must be one of the evidence ids")
    if not isinstance(quote, str) or not quote.strip() or len(quote) > MAX_QUOTE_CHARS:
        raise ProgramError(
            f"Step #{step}: quote must be a non-empty exact span of at most "
            f"{MAX_QUOTE_CHARS} characters"
        )
    text = citations[citation_id].quote
    mentions = extract_numbers(text)
    candidates: dict[tuple[Decimal, Decimal, str | None], NumberMention] = {}
    position = text.find(quote)
    while position >= 0:
        inside = [
            mention
            for mention in mentions
            if position <= mention.core_start and mention.core_end <= position + len(quote)
        ]
        if len(inside) > 1:
            raise ProgramError(
                f"Step #{step}: quote contains more than one number; quote only the value span"
            )
        if inside:
            mention = inside[0]
            candidates[(mention.value, mention.scale, mention.unit)] = mention
        position = text.find(quote, position + 1)
    if not candidates:
        raise ProgramError(
            f"Step #{step}: quote is not an exact span of {citation_id} containing one complete "
            "number; copy the characters exactly from the evidence text"
        )
    if len(candidates) > 1:
        raise ProgramError(
            f"Step #{step}: quote is ambiguous in {citation_id}; include more surrounding words"
        )
    mention = next(iter(candidates.values()))
    return _Value(
        value=mention.operand_value,
        operand=CalculationOperand(
            kind="evidence",
            value=format_decimal(mention.operand_value),
            citation_id=citation_id,
            quote=quote,
            source_value=mention.text[:500],
            scale=mention.scale_word,
            unit=mention.unit,
        ),
        citation_ids=frozenset({citation_id}),
        unit=_evidence_unit(text, mention),
        evidence_backed=True,
    )


def _operand(
    raw: Any, step: int, results: Sequence[_Value], citations: Mapping[str, Citation]
) -> _Value:
    if not isinstance(raw, Mapping):
        raise ProgramError(f"Step #{step}: each argument must be an object")
    keys = set(raw)
    if keys == {"citation_id", "quote"}:
        return _bind_evidence(raw, step, citations)
    if keys == {"ref"}:
        ref = raw["ref"]
        if not isinstance(ref, str) or not re.fullmatch(r"#\d{1,2}", ref):
            raise ProgramError(f"Step #{step}: ref must look like '#0'")
        index = int(ref[1:])
        if index >= step:
            raise ProgramError(f"Step #{step}: ref may only point to an earlier step")
        prior = results[index]
        return _Value(
            value=prior.value,
            operand=CalculationOperand(kind="step", value=format_decimal(prior.value), step=index),
            citation_ids=prior.citation_ids,
            unit=prior.unit,
            evidence_backed=prior.evidence_backed,
        )
    if keys == {"constant"}:
        constant = raw["constant"]
        if isinstance(constant, bool) or not isinstance(constant, (int, float, str)):
            raise ProgramError(f"Step #{step}: constant must be a number")
        try:
            value = Decimal(str(constant))
        except InvalidOperation:
            raise ProgramError(f"Step #{step}: constant must be a number") from None
        # sNaN cannot even be hashed for the membership test below.
        if not value.is_finite() or value not in CONSTANTS:
            raise ProgramError(
                f"Step #{step}: unbound literal; constants are limited to "
                "1, 2, 4, 12, 100, 1000 and 1000000. Bind other numbers to evidence with "
                "citation_id and an exact quote"
            )
        return _Value(
            value=value,
            operand=CalculationOperand(kind="constant", value=format_decimal(value)),
            citation_ids=frozenset(),
            unit=_CONSTANT,
            evidence_backed=False,
        )
    raise ProgramError(
        f"Step #{step}: argument must be exactly one of {{citation_id, quote}}, {{ref}} or "
        "{constant}"
    )


_SCALING_OPS = frozenset({"add", "subtract", "multiply", "divide", "exp"})


def _mix_error(op: str, step: int) -> ProgramError:
    return ProgramError(
        f"Step #{step}: {op} cannot mix percentages or ratios with amounts; convert first"
    )


def _currency(units: Sequence[_Unit], op: str, step: int) -> str | None:
    tokens = {unit.currency for unit in units if unit.currency is not None}
    if len(tokens) > 1:
        raise ProgramError(
            f"Step #{step}: {op} cannot mix amounts in different currencies; currencies are "
            "never converted"
        )
    return next(iter(tokens), None)


def _finest_scale(units: Sequence[_Unit]) -> UnitScale | None:
    scales = [scale for scale in (unit.scale for unit in units) if scale is not None]
    if len(scales) < len(units):
        return None
    finest: UnitScale = min(scales, key=lambda scale: _SCALES[scale])
    return finest


def _single_scale(*units: _Unit) -> UnitScale | None:
    scales = [scale for scale in (unit.scale for unit in units) if scale is not None]
    return scales[0] if len(scales) == 1 else None


def _combine(units: Sequence[_Unit], op: str, step: int) -> _Unit:
    """Unit of an additive combination (sum, difference, comparison or table op)."""
    dims = {unit.dim for unit in units}
    if dims <= {"ratio"}:
        percent = all(unit.display == "percent" for unit in units)
        return _Unit("ratio", display="percent" if percent else "ratio")
    if dims == {"percent"}:
        return _PERCENT
    if not dims <= {"currency", "per_share", "count", "unitless"}:
        raise _mix_error(op, step)
    currency = _currency(units, op, step)
    if "count" in dims and dims & _MONEY:
        raise ProgramError(
            f"Step #{step}: {op} cannot mix counts such as shares with currency amounts"
        )
    if "per_share" in dims:
        if any(unit.scale for unit in units):
            raise ProgramError(
                f"Step #{step}: {op} cannot mix per-share amounts with scaled totals"
            )
        return _Unit("per_share", currency=currency)
    if "currency" in dims:
        return _Unit("currency", currency=currency, scale=_finest_scale(units))
    if "count" in dims:
        shares = all(unit.shares for unit in units if unit.dim == "count")
        return _Unit("count", scale=_finest_scale(units), shares=shares)
    return _Unit("unitless", scale=_finest_scale(units))


def _power(base: _Unit, exponent: _Unit | None, step: int) -> _Unit:
    if base.dim in _MONEY or (exponent is not None and exponent.dim in _MONEY):
        raise ProgramError(
            f"Step #{step}: exp needs a ratio or unitless base and exponent; currency "
            "amounts cannot be raised to a power"
        )
    if base.dim == "ratio":
        return _Unit("ratio", display="ratio")
    if base.dim == "percent":
        return _PERCENT
    return _Unit("unitless")


def _product(left: _Unit, right: _Unit, step: int) -> _Unit:
    if left.dim == "ratio" and right.dim == "ratio":
        percent = left.display == right.display == "percent"
        return _Unit("ratio", display="percent" if percent else "ratio")
    if right.dim == "ratio":
        return left
    if left.dim == "ratio":
        return right
    if left.dim in _MONEY and right.dim in _MONEY:
        raise ProgramError(f"Step #{step}: multiply cannot multiply two currency amounts")
    money, other = (left, right) if left.dim in _MONEY else (right, left)
    if money.dim in _MONEY:
        # A per-share amount times a share count (or table quantity) is a total.
        return _Unit("currency", currency=money.currency, scale=_single_scale(money, other))
    if {left.dim, right.dim} == {"count", "unitless"}:
        count = left if left.dim == "count" else right
        return _Unit("count", scale=_single_scale(left, right), shares=count.shares)
    return _Unit("unitless", scale=_single_scale(left, right))


def _quotient(left: _Unit, right: _Unit, step: int) -> _Unit:
    if right.dim == "ratio":
        return _Unit("ratio", display="ratio") if left.dim == "ratio" else left  # amount / rate
    if left.dim == "ratio":
        raise _mix_error("divide", step)
    currency = _currency([left, right], "divide", step)
    if left.dim == "currency" and right.dim == "count":
        return _Unit("per_share" if right.shares else "currency", currency=currency)
    if left.dim == "currency" and right.dim == "per_share":
        return _Unit("count", shares=True)
    return _Unit("ratio", display="ratio")


def _result_unit(op: str, operands: Sequence[_Value], step: int) -> _Unit:
    """Unit of a step result; also enforces evidence lineage, constant roles and units."""
    constants = [operand for operand in operands if operand.unit.dim == "constant"]
    bound = [operand for operand in operands if operand.unit.dim != "constant"]
    if not any(operand.evidence_backed for operand in bound):
        raise ProgramError(
            f"Step #{step}: every step needs an evidence-bound argument (a cited quote or a "
            "ref to an evidence-based step); constants cannot create values on their own"
        )
    if any(operand.unit.dim == "boolean" for operand in operands):
        raise ProgramError(f"Step #{step}: a 'greater' result is a yes/no flag, not a number")
    if constants and (len(constants) > 1 or op not in _SCALING_OPS):
        raise ProgramError(
            f"Step #{step}: a constant may only be the single scaling or offset argument "
            "of add, subtract, multiply, divide or exp"
        )
    units = [operand.unit for operand in bound]
    if op == "greater":
        _combine(units, op, step)
        return _BOOLEAN
    if op == "percent_change":
        _combine(units, op, step)
        return _PERCENT
    if constants:
        constant, other = constants[0].value, units[0]
        if op in {"divide", "exp"} and operands[1].unit.dim != "constant":
            raise ProgramError(f"Step #{step}: in {op} a constant may only be the second argument")
        if op in {"add", "subtract"}:
            if constant != 1 or other.dim != "ratio":
                raise ProgramError(
                    f"Step #{step}: add/subtract only accept the constant 1 next to a "
                    "percentage or ratio (for example 1 + growth rate)"
                )
            return _Unit("ratio", display="ratio")
        if op == "multiply" and constant == 100 and other.dim == "ratio":
            return _PERCENT
        if op == "divide" and constant == 100 and other.dim == "percent":
            return _Unit("ratio", display="percent")
        if op == "exp":
            return _power(other, None, step)
        return other
    if op in {"add", "subtract"} or op in TABLE_OPS:
        return _combine(units, op, step)
    if op == "exp":
        return _power(units[0], units[1], step)
    if any(unit.dim == "percent" for unit in units):
        raise ProgramError(
            f"Step #{step}: percent_change results are in percent; divide by 100 before "
            f"using them in {op}"
        )
    left, right = units
    return _product(left, right, step) if op == "multiply" else _quotient(left, right, step)


def _apply(op: str, values: list[Decimal], step: int) -> Decimal:
    if op in BINARY_OPS and len(values) != 2:
        raise ProgramError(f"Step #{step}: {op} takes exactly two arguments")
    if op in TABLE_OPS and not 1 <= len(values) <= MAX_ARGS:
        raise ProgramError(f"Step #{step}: {op} takes between 1 and {MAX_ARGS} arguments")
    try:
        if op == "add":
            return values[0] + values[1]
        if op == "subtract":
            return values[0] - values[1]
        if op == "multiply":
            return values[0] * values[1]
        if op == "divide":
            if values[1] == 0:
                raise ProgramError(f"Step #{step}: division by zero")
            return values[0] / values[1]
        if op == "percent_change":
            if values[0] == 0:
                raise ProgramError(f"Step #{step}: percent_change from zero is undefined")
            return (values[1] - values[0]) / values[0] * 100
        if op == "greater":
            return Decimal(1) if values[0] > values[1] else Decimal(0)
        if op == "exp":
            base, exponent = values
            if abs(exponent) > MAX_EXPONENT:
                raise ProgramError(f"Step #{step}: exponent magnitude is limited to {MAX_EXPONENT}")
            if exponent != exponent.to_integral_value() and base <= 0:
                raise ProgramError(f"Step #{step}: fractional exponent needs a positive base")
            if base == 0 and exponent < 0:
                raise ProgramError(f"Step #{step}: division by zero")
            if abs(base) > Decimal("1e6"):
                raise ProgramError(f"Step #{step}: exp base magnitude is limited to 1e6")
            return base**exponent
        if op == "table_sum":
            return sum(values, Decimal(0))
        if op == "table_average":
            return sum(values, Decimal(0)) / len(values)
        if op == "table_max":
            return max(values)
        if op == "table_min":
            return min(values)
    except (InvalidOperation, DivisionByZero, Overflow):
        raise ProgramError(f"Step #{step}: arithmetic is undefined for these operands") from None
    raise ProgramError(f"Step #{step}: unsupported op; use one of {', '.join(OPS)}")


def run_program(arguments: Any, citations: Sequence[Citation]) -> Calculation:
    """Validate and execute a program; raises ProgramError with a model-correctable message."""
    if not isinstance(arguments, Mapping) or set(arguments) != {"steps"}:
        raise ProgramError(
            "calculator expects {'steps': [...]}; free-form expressions are not accepted. "
            "Each step is {'op': ..., 'args': [...]} with evidence-bound arguments"
        )
    steps = arguments["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_STEPS:
        raise ProgramError(f"steps must be a list of 1 to {MAX_STEPS} steps")
    total_args = sum(
        len(step["args"])
        for step in steps
        if isinstance(step, Mapping) and isinstance(step.get("args"), list)
    )
    if total_args > MAX_TOTAL_ARGS:
        raise ProgramError(f"program exceeds {MAX_TOTAL_ARGS} total arguments")
    by_id = {citation.id: citation for citation in citations}
    results: list[_Value] = []
    trace: list[CalculationStep] = []
    with localcontext(_context()) as context:
        context.traps[InvalidOperation] = True
        context.traps[DivisionByZero] = True
        context.traps[Overflow] = True
        for index, raw in enumerate(steps):
            if not isinstance(raw, Mapping) or set(raw) != {"op", "args"}:
                raise ProgramError(f"Step #{index} must be exactly {{'op': ..., 'args': [...]}}")
            op, args = raw["op"], raw["args"]
            if not isinstance(op, str) or op not in OPS:
                raise ProgramError(f"Step #{index}: unsupported op; use one of {', '.join(OPS)}")
            if not isinstance(args, list) or not 1 <= len(args) <= MAX_ARGS:
                raise ProgramError(f"Step #{index}: args must be a list of 1 to {MAX_ARGS} items")
            try:
                operands = [_operand(arg, index, results, by_id) for arg in args]
            except ProgramError:
                raise
            except (TypeError, ValueError, ArithmeticError, KeyError, IndexError):
                raise ProgramError(f"Step #{index}: an argument could not be parsed") from None
            if op in BINARY_OPS and len(operands) != 2:
                raise ProgramError(f"Step #{index}: {op} takes exactly two arguments")
            unit = _result_unit(op, operands, index)
            value = _bounded(_apply(op, [operand.value for operand in operands], index), index)
            citation_ids = frozenset().union(*(operand.citation_ids for operand in operands))
            results.append(
                _Value(
                    value=value,
                    operand=CalculationOperand(
                        kind="step", value=format_decimal(value), step=index
                    ),
                    citation_ids=citation_ids,
                    unit=unit,
                    evidence_backed=True,
                )
            )
            trace.append(
                CalculationStep(
                    index=index,
                    op=op,
                    operands=[operand.operand for operand in operands],
                    result=format_decimal(value),
                    kind=_public_kind(unit.kind),
                    citation_ids=sorted(citation_ids, key=lambda value: int(value[1:])),
                    evidence_backed=True,
                    unit=_public_unit(unit),
                )
            )
    if not results[-1].citation_ids:
        raise ProgramError("program must use at least one evidence-bound argument")
    return Calculation(
        steps=trace,
        result=trace[-1].result,
        citation_ids=sorted(results[-1].citation_ids, key=lambda value: int(value[1:])),
        unit=trace[-1].unit,
    )


def _public_kind(kind: _Kind) -> CalculationKind:
    if kind == "constant":  # unreachable: _result_unit never returns a bare constant
        raise ProgramError("constants cannot be step results")
    return kind


_UNIT_KINDS: dict[_Dim, UnitKind] = {
    "currency": "currency",
    "per_share": "per_share",
    "count": "count",
    "unitless": "unitless",
    "ratio": "ratio",
    "percent": "percent",
    "boolean": "unitless",
}


def _public_unit(unit: _Unit) -> CalculationUnit:
    if unit.dim == "constant":  # unreachable, as above
        raise ProgramError("constants cannot be step results")
    money = unit.dim in _MONEY
    iso = unit.currency if unit.currency and _ISO_CURRENCY.fullmatch(unit.currency) else None
    return CalculationUnit(
        kind=_UNIT_KINDS[unit.dim],
        currency=iso if money else None,
        scale=unit.scale if unit.dim in {"currency", "count", "unitless"} else None,
        display=unit.display,
    )


def program_tool_result(calculation: Calculation) -> dict[str, Any]:
    return {
        "result": calculation.result,
        "citation_ids": calculation.citation_ids,
        "steps": [
            {
                "step": f"#{step.index}",
                "op": step.op,
                "inputs": [operand.value for operand in step.operands],
                "result": step.result,
                "kind": step.kind,
                "unit": step.unit.model_dump(exclude_none=True) if step.unit else None,
            }
            for step in calculation.steps
        ],
        "note": (
            "Amounts are in base units (scale words applied); percentages are fractions. "
            "Units are derived from the evidence; state the currency they show. "
            "Cite the evidence ids used by the program."
        ),
    }
