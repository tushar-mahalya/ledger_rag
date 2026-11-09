"""The raw-model contract: the same evidence, the model answers on its own.

This is the control arm for every Ledger comparison. The reader sees the same page
(or table and paragraphs) and question as the Ledger pipeline but no exemplars, no
program language, no grounding and no voting: it may reason in free text and ends
with ``ANSWER:`` (and ``SCALE:`` for TAT-QA). Answers are parsed leniently so a
correct answer is never lost to formatting: ``$1,234``, ``16.4%`` and ``(5)`` all
read as numbers (a percent sign divides by 100, as the official FinQA executor does).
"""

import re
from dataclasses import dataclass

from ledger_rag.reasoning.consistency import Candidate
from ledger_rag.reasoning.grounding import Evidence
from ledger_rag.reasoning.solver import Problem
from ledger_rag.reasoning.typed import SCALES, outcome

FINQA_PROMPT = """You answer questions about a page of a company's annual report.
Use the page; you may reason step by step. End your reply with exactly one line:
ANSWER: <the final answer: a number, or yes/no>"""

TATQA_PROMPT = """You answer questions about a table and paragraphs from a company's annual report.
Use them; you may reason step by step. End your reply with exactly these two lines:
ANSWER: <the final answer; separate several items with " | ">
SCALE: <none|thousand|million|billion|percent>"""

_ANSWER = re.compile(r"^\s*(?:\*\*)?ANSWER(?:\*\*)?\s*:\s*(.+?)\s*$", re.I | re.M)
_SCALE = re.compile(r"^\s*(?:\*\*)?SCALE(?:\*\*)?\s*:\s*(.*?)\s*$", re.I | re.M)
_BARE = re.compile(r"\(?-?\$?\s*-?(?:\d[\d,]*(?:\.\d+)?|\.\d+)\)?")
_NUMERIC = re.compile(r"^\(?-?\$?\s*-?\d[\d,]*(?:\.\d+)?\s*%?\)?$|^\(?-?\$?\s*-?\.\d+\s*%?\)?$")


def _last(pattern: re.Pattern[str], text: str) -> str | None:
    found = pattern.findall(text)
    return found[-1].strip().strip("`*").strip() if found else None


def read_number(text: str) -> float | None:
    """``$1,234`` -> 1234.0, ``16.4%`` -> 0.164, ``(5)`` -> -5.0; None if not numeric."""
    raw = text.strip().rstrip(".")
    raw = re.sub(r"\s*(?:million|billion|thousand|dollars|usd)$", "", raw, flags=re.I)
    if not _NUMERIC.match(raw):
        return None
    negative = raw.startswith("(") and raw.endswith(")") or raw.lstrip("($ ").startswith("-")
    digits = re.sub(r"[^\d.]", "", raw)
    try:
        value = float(digits)
    except ValueError:
        return None
    if "%" in raw:
        value /= 100
    return -value if negative else value


@dataclass(frozen=True)
class DirectNumberTask:
    """FinQA/ConvFinQA raw arm: a number or yes/no read from ``ANSWER:``."""

    system_prompt: str = FINQA_PROMPT

    def judge(self, text: str, problem: Problem, evidence: Evidence) -> Candidate:
        answer = _last(_ANSWER, text)
        if answer is None:
            return Candidate(None, None, "parse")
        lowered = answer.lower().rstrip(".")
        if lowered in {"yes", "no"}:
            return Candidate(None, lowered, "ok")
        value = read_number(answer)
        if value is None:
            return Candidate(None, None, "parse")
        return Candidate(None, round(value, 5), "ok")


@dataclass(frozen=True)
class DirectTypedTask:
    """TAT-QA raw arm: an answer string (or items) with a declared scale."""

    system_prompt: str = TATQA_PROMPT

    def judge(self, text: str, problem: Problem, evidence: Evidence) -> Candidate:
        answer = _last(_ANSWER, text)
        if not answer:
            return Candidate(None, None, "parse")
        raw_scale = (_last(_SCALE, text) or "").lower().rstrip(".")
        scale = "" if raw_scale in {"", "none", "n/a", "-"} else raw_scale
        if scale in {"%", "percentage"}:
            scale = "percent"
        if scale.endswith("s") and scale[:-1] in SCALES:
            scale = scale[:-1]
        if scale not in SCALES:
            scale = ""
        items = [part.strip().strip("\"'") for part in answer.split("|") if part.strip()]
        # Only a bare amount becomes a number; "$85.1 million" or "36%" stay text so
        # the official metric compares them with their own units.
        if len(items) == 1 and _BARE.fullmatch(items[0]):
            number = read_number(items[0])
            if number is not None:
                return Candidate(None, outcome("arithmetic", [number], scale), "ok")
        kind = "multi-span" if len(items) > 1 else "span"
        return Candidate(None, outcome(kind, items, scale), "ok")
