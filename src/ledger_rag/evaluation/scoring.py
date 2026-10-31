"""Structured final-answer parsing and FinQA-style answer scoring.

Answers are scored only from an explicit ``ANSWER:`` line, never from the first
number found in free text. Two primary metrics are reported separately:

- ``exact``: numeric equality after rounding the prediction and the gold program
  output to the precision stated by the gold answer string; normalized
  case/punctuation-insensitive equality for yes/no and text answers.
- ``tolerant``: ``exact`` or a relative error within ``rel_tol`` (default 1%).

A percentage is converted to a ratio only when the prediction carries an explicit
``%`` sign. Matching ``gold * 100`` or ``gold / 100`` without a unit is reported as a
separately named diagnostic (``percent_scale_diagnostic``), never as accuracy.

Reasoning models may leak ``<think>...</think>`` text into the answer content (for
example when a server runs without a reasoning parser). That text is removed before
parsing, so an ``ANSWER:`` line inside reasoning is never scored.
"""

import math
import re
from dataclasses import dataclass, replace
from decimal import ROUND_HALF_UP, Decimal

from ledger_rag.generation.reasoning import split_reasoning

ABSTAIN_TOKENS = frozenset({"insufficient", "unknown", "abstain", "n/a", "cannot determine"})
_FIELD = re.compile(r"^\s*\**\s*(ANSWER|CONFIDENCE|CITATIONS)\s*\**\s*:\s*(.*?)\s*$", re.I | re.M)
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
# Scale words are tolerated as trailing labels but never applied as multipliers.
_UNIT_WORDS = re.compile(
    r"\b(?:usd|dollars?|millions?|billions?|thousands?|mm|bn|basis points|bps|percent|"
    r"per ?cent|times|x|shares?|units?)\b",
    re.I,
)
_CITATION = re.compile(r"\bC(\d+)\b", re.I)


@dataclass(frozen=True)
class StructuredAnswer:
    """Fields parsed from the final structured lines of a model response."""

    raw: str | None
    value: float | None
    text: str | None
    percent: bool
    abstained: bool
    confidence: float | None
    citations: tuple[int, ...]
    format_error: str | None
    think_leak: bool = False


def _clean_number(field: str) -> tuple[float | None, bool, str | None]:
    """Parse one numeric value; returns (value, explicit_percent, error)."""
    text = field.strip().strip("`*").strip()
    text = text.replace("\u2212", "-").replace("\u2013", "-").replace(",", "")
    text = re.sub(r"\[C\d+\]", " ", text, flags=re.I)
    negative_parens = bool(re.fullmatch(r"\(\s*[^()]*\)\s*%?", text))
    if negative_parens:
        text = text.replace("(", "").replace(")", "")
    percent = "%" in text or bool(re.search(r"\bper ?cent\b", text, re.I))
    stripped = _UNIT_WORDS.sub(" ", text.replace("%", " ").replace("$", " "))
    stripped = stripped.strip().rstrip(".").strip()
    matches = _NUMBER.findall(stripped)
    if len(matches) != 1:
        return None, percent, "no_number" if not matches else "multiple_numbers"
    residue = _NUMBER.sub("", stripped, count=1).strip()
    if residue and residue not in {"-", "+"}:
        return None, percent, "unparsed_text"
    try:
        value = float(matches[0])
    except ValueError:
        return None, percent, "no_number"
    if not math.isfinite(value):
        return None, percent, "non_finite"
    if negative_parens:
        value = -abs(value)
    return value, percent, None


def parse_structured(text: str) -> StructuredAnswer:
    """Parse the last ``ANSWER:``, ``CONFIDENCE:`` and ``CITATIONS:`` lines.

    Leaked reasoning is stripped first (see ``split_reasoning``) and flagged.
    """
    answer, leaked = split_reasoning(text)
    parsed = _parse_fields(answer)
    return replace(parsed, think_leak=True) if leaked else parsed


def _parse_fields(text: str) -> StructuredAnswer:
    fields: dict[str, str] = {}
    for match in _FIELD.finditer(text):
        fields[match.group(1).upper()] = match.group(2)
    confidence: float | None = None
    if "CONFIDENCE" in fields:
        found = _NUMBER.search(fields["CONFIDENCE"])
        if found:
            stated = float(found.group())
            scaled = stated / 100 if stated > 1 else stated
            confidence = scaled if 0 <= scaled <= 1 else None
    citations = tuple(sorted({int(n) for n in _CITATION.findall(fields.get("CITATIONS", ""))}))
    raw = fields.get("ANSWER")
    if raw is None:
        return StructuredAnswer(None, None, None, False, False, confidence, citations, "missing")
    normalized = normalize_text(raw)
    if not normalized:
        return StructuredAnswer(raw, None, None, False, False, confidence, citations, "empty")
    if normalized in ABSTAIN_TOKENS or normalized.startswith("insufficient"):
        return StructuredAnswer(raw, None, None, False, True, confidence, citations, None)
    if normalized in {"yes", "no"}:
        return StructuredAnswer(raw, None, normalized, False, False, confidence, citations, None)
    value, percent, error = _clean_number(raw)
    if value is None:
        # Free text is kept for text-answer comparison but is not a numeric answer.
        return StructuredAnswer(raw, None, normalized, False, False, confidence, citations, error)
    return StructuredAnswer(raw, value, None, percent, False, confidence, citations, None)


def normalize_text(text: str) -> str:
    text = re.sub(r"\[C\d+\]", " ", text, flags=re.I)
    text = re.sub(r"[^\w%./\- ]", " ", text.casefold())
    return " ".join(text.split()).strip(" .")


def _decimals(text: str) -> int:
    match = re.search(r"\.(\d+)", text)
    return len(match.group(1)) if match else 0


def _round(value: float, places: int) -> Decimal:
    quantum = Decimal(1).scaleb(-places)
    return Decimal(repr(value)).quantize(quantum, rounding=ROUND_HALF_UP)


def gold_precision(gold: float, stated: str) -> tuple[int, str]:
    """Decimal places on the ``exe_ans`` scale implied by the gold answer string.

    FinQA renders percentages as ``12.6%`` while ``exe_ans`` is ``0.12574``; the
    stated precision is shifted by two places in that case. If the stated answer
    is missing or inconsistent with the program output, fall back to three
    significant figures and say so.
    """
    cleaned = stated.strip().replace(",", "").replace("$", "")
    match = _NUMBER.search(cleaned)
    if match:
        decimals = _decimals(match.group())
        number = float(match.group())
        candidates = [(number, decimals)]
        if "%" in cleaned:
            candidates.insert(0, (number / 100, decimals + 2))
        else:
            candidates.append((number / 100, decimals + 2))
        for value, places in candidates:
            if places >= 0 and _round(gold, places) == _round(value, places):
                return places, "stated"
    if gold == 0:
        return 2, "fallback_3sf"
    return max(0, 2 - math.floor(math.log10(abs(gold)))), "fallback_3sf"


@dataclass(frozen=True)
class Score:
    exact: bool
    tolerant: bool
    percent_scale_diagnostic: bool
    precision: int | None
    precision_source: str | None


def score(
    answer: StructuredAnswer,
    gold: float | str,
    stated: str = "",
    *,
    rel_tol: float = 0.01,
) -> Score:
    """Score a parsed answer against a FinQA ``exe_ans`` gold value."""
    if answer.abstained:
        return Score(False, False, False, None, None)
    if isinstance(gold, str):
        target = normalize_text(gold)
        predicted = answer.text if answer.text is not None else normalize_text(answer.raw or "")
        same = bool(predicted) and predicted == target
        return Score(same, same, False, None, None)
    places, source = gold_precision(gold, stated)
    if answer.value is None:
        return Score(False, False, False, places, source)
    value = answer.value / 100 if answer.percent else answer.value
    exact = _round(value, places) == _round(gold, places)
    close = math.isclose(value, gold, rel_tol=rel_tol, abs_tol=1e-9)
    diagnostic = exact or close
    if not diagnostic:
        for alternative in (answer.value, answer.value / 100, answer.value * 100):
            if math.isclose(alternative, gold, rel_tol=rel_tol, abs_tol=1e-9):
                diagnostic = True
                break
    return Score(exact, exact or close, diagnostic, places, source)
