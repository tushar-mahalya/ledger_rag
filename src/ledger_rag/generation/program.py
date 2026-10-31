"""Evidence-bound arithmetic programs (FinQA-style DSL) and financial number normalization.

Every operand is bound to evidence: an exact quoted span of a cited passage, a prior
step result (``#0`` is the first step, as in FinQA) or a small whitelisted constant.
Free numeric literals are rejected so the model cannot introduce unsupported inputs.
Arithmetic uses ``Decimal`` with 28 significant digits and ROUND_HALF_EVEN; results are
displayed with at most 10 decimal places.
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

from ledger_rag.core.models import Calculation, CalculationOperand, CalculationStep, Citation

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


def extract_numbers(text: str, *, include_years: bool = True) -> list[NumberMention]:
    """Normalize financial numbers in text; offsets refer to the original string.

    Parentheses mark a negative only around a bare amount, e.g. ``(1,234)`` or
    ``$(1,234)``; prose parentheticals with scale words or percentages stay positive.
    """
    mentions: list[NumberMention] = []
    for match in _NUMBER.finditer(_mask(text)):
        core = match.group("core")
        unit_text = re.sub(r"\s+", " ", (match.group("unit") or "").strip().lower())
        bracketed = bool(match.group("open") and match.group("close"))
        inner_scale = (match.group("scale") or "").strip().lower()
        outer_scale = (match.group("scale2") or "").strip().lower() if bracketed else ""
        scale_key = inner_scale or outer_scale
        currency = bool(match.group("currency") or match.group("lead"))
        if scale_key in {"k", "m", "b"} and not currency:
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
            )
        )
    return mentions


@dataclass(frozen=True)
class _Value:
    value: Decimal
    operand: CalculationOperand
    citation_ids: frozenset[str]


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
            source_value=mention.text,
            scale=mention.scale_word,
            unit=mention.unit,
        ),
        citation_ids=frozenset({citation_id}),
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
        )
    if keys == {"constant"}:
        constant = raw["constant"]
        if isinstance(constant, bool) or not isinstance(constant, (int, float, str)):
            raise ProgramError(f"Step #{step}: constant must be a number")
        try:
            value = Decimal(str(constant))
        except InvalidOperation:
            raise ProgramError(f"Step #{step}: constant must be a number") from None
        if value not in CONSTANTS:
            raise ProgramError(
                f"Step #{step}: unbound literal; constants are limited to "
                "1, 2, 4, 12, 100, 1000 and 1000000. Bind other numbers to evidence with "
                "citation_id and an exact quote"
            )
        return _Value(
            value=value,
            operand=CalculationOperand(kind="constant", value=format_decimal(value)),
            citation_ids=frozenset(),
        )
    raise ProgramError(
        f"Step #{step}: argument must be exactly one of {{citation_id, quote}}, {{ref}} or "
        "{constant}"
    )


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
            operands = [_operand(arg, index, results, by_id) for arg in args]
            value = _bounded(_apply(op, [operand.value for operand in operands], index), index)
            citation_ids = frozenset().union(*(operand.citation_ids for operand in operands))
            results.append(
                _Value(
                    value=value,
                    operand=CalculationOperand(
                        kind="step", value=format_decimal(value), step=index
                    ),
                    citation_ids=citation_ids,
                )
            )
            trace.append(
                CalculationStep(
                    index=index,
                    op=op,
                    operands=[operand.operand for operand in operands],
                    result=format_decimal(value),
                )
            )
    if not results[-1].citation_ids:
        raise ProgramError("program must use at least one evidence-bound argument")
    return Calculation(
        steps=trace,
        result=trace[-1].result,
        citation_ids=sorted(results[-1].citation_ids, key=lambda value: int(value[1:])),
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
            }
            for step in calculation.steps
        ],
        "note": (
            "Amounts are in base units (scale words applied); percentages are fractions. "
            "Cite the evidence ids used by the program."
        ),
    }
