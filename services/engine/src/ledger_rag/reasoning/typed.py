"""Typed answers for hybrid table-text QA (TAT-QA): spans, counts or programs, plus scale.

The reader declares an answer type and the answer's scale. Spans must be copied
from the evidence (checked after normalization), counts are integers, and
arithmetic is a program that a deterministic executor runs, exactly as in
:mod:`.solver`. Candidates vote on a canonical JSON outcome, so two samples agree
when they give the same normalized answer at the same scale.

:func:`expression_program` turns annotators' arithmetic derivations
(``(16,284 - 6,509) / 6,509``) into DSL programs so training problems can serve
as exemplars; callers keep only conversions that reproduce the gold answer.
"""

import ast
import json
import re
import string
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ledger_rag.reasoning.consistency import Candidate
from ledger_rag.reasoning.grounding import Evidence
from ledger_rag.reasoning.solver import Executor, Problem, run_program

TYPES = ("span", "multi-span", "count", "arithmetic")
SCALES = ("", "thousand", "million", "billion", "percent")
SYSTEM_PROMPT = """You answer questions about a table and paragraphs from a company's annual report.
First decide the answer type, then answer in the matching way:
- span: one piece of text copied exactly from the table or paragraphs -> ANSWER
- multi-span: several pieces copied exactly, separated by " | " -> ANSWER
- count: how many items -> ANSWER (an integer)
- arithmetic: a calculation -> PROGRAM. Never calculate yourself; a calculator executes it.

Program operations, each with exactly two operands:
  add(a, b)  subtract(a, b)  multiply(a, b)  divide(a, b)  exp(a, b)  greater(a, b)
Operands: a number copied exactly from the table or paragraphs (no $, % or thousands
separators; write (1,234) as -1234), a constant (const_2, const_100, const_1000,
const_m1 for -1) or #k, the result of step k (#0 is the first step). The last step is
the answer.

SCALE is the unit of the answer: none, thousand, million, billion or percent.
Amounts keep the unit the table or text states (a table "in millions" -> million).
A percentage answer is in percent: multiply a ratio by const_100 and use SCALE: percent.

Reply in exactly this format and nothing after it:
FACTS:
- <each figure or text you use>
TYPE: <span|multi-span|count|arithmetic>
SCALE: <none|thousand|million|billion|percent>
PROGRAM: <steps separated by ", ">   (arithmetic only)
ANSWER: <answer>                     (span, multi-span or count only)"""

_FIELD = re.compile(r"^\s*(?:\*\*)?(TYPE|SCALE|ANSWER)(?:\*\*)?\s*:\s*(.*)$", re.I | re.M)
_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_NUMBER_TOKEN = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?|(?<![\w.])-?\.\d+%?")


def normalize(text: str) -> str:
    """Lowercase, drop punctuation and articles, squeeze whitespace (DROP-style)."""
    lowered = _ARTICLES.sub(" ", _PUNCT.sub(" ", text.lower()))
    return " ".join(lowered.split())


def _fields(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for match in _FIELD.finditer(text):
        found[match.group(1).upper()] = match.group(2).strip().strip("`*").strip()
    return found


def _scale(raw: str) -> str | None:
    value = raw.strip().lower().rstrip(".")
    if value in {"", "none", "n/a", "-", "no scale"}:
        return ""
    for scale in SCALES[1:]:
        if value in {scale, scale + "s"} or (scale == "percent" and value in {"%", "percentage"}):
            return scale
    return None


def outcome(kind: str, answer: Sequence[str | float], scale: str) -> str:
    """Canonical voting key and stored value: JSON of type, answer list and scale."""
    if kind == "arithmetic":
        rendered: list[str] = [f"{round(float(answer[0]), 2):.2f}"]
    elif kind == "count":
        rendered = [str(int(float(answer[0])))]
    else:
        rendered = sorted(str(item).strip() for item in answer)
    return json.dumps({"type": kind, "answer": rendered, "scale": scale}, sort_keys=True)


def decode(value: str) -> dict[str, Any]:
    decoded: dict[str, Any] = json.loads(value)
    return decoded


@dataclass(frozen=True)
class TypedAnswerTask:
    """TAT-QA contract; arithmetic programs run on the injected executor."""

    executor: Executor
    system_prompt: str = SYSTEM_PROMPT

    def judge(self, text: str, problem: Problem, evidence: Evidence) -> Candidate:
        fields = _fields(text)
        kind = fields.get("TYPE", "").lower().replace("_", "-").strip()
        scale = _scale(fields.get("SCALE", ""))
        if kind not in TYPES or scale is None:
            return Candidate(None, None, "parse")
        if kind == "arithmetic":
            ran = run_program(text, problem, evidence, self.executor)
            if ran.value is None or isinstance(ran.value, str):
                return Candidate(ran.program, None, ran.reason if ran.value is None else "type")
            return Candidate(ran.program, outcome(kind, [ran.value], scale), "ok")
        answer = fields.get("ANSWER", "").strip()
        if not answer:
            return Candidate(None, None, "parse")
        if kind == "count":
            if not re.fullmatch(r"\d+", answer):
                return Candidate(None, None, "parse")
            return Candidate(None, outcome(kind, [answer], scale), "ok")
        spans = [part.strip().strip("\"'") for part in answer.split("|") if part.strip()]
        haystack = normalize(" ".join([*problem.pre_text, *problem.post_text, *_cells(problem)]))
        if not spans or any(normalize(span) not in haystack for span in spans):
            return Candidate(None, None, "ungrounded")
        return Candidate(None, outcome(kind, spans, scale), "ok")


def _cells(problem: Problem) -> list[str]:
    return [cell for row in problem.table for cell in row]


class _Emit:
    def __init__(self) -> None:
        self.steps: list[str] = []

    def add(self, op: str, left: str, right: str) -> str:
        self.steps.append(f"{op}({left}, {right})")
        return f"#{len(self.steps) - 1}"


_OPS = {ast.Add: "add", ast.Sub: "subtract", ast.Mult: "multiply", ast.Div: "divide"}


def _walk(node: ast.expr, names: dict[str, str], emit: _Emit) -> str:
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return f"const_{node.value}" if node.value in (2, 100, 1000) else str(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _walk(node.operand, names, emit)
        if inner.startswith("#") or inner.startswith("const_"):
            return emit.add("multiply", inner, "const_m1")
        return inner[1:] if inner.startswith("-") else "-" + inner
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _walk(node.left, names, emit)
        right = _walk(node.right, names, emit)
        return emit.add(_OPS[type(node.op)], left, right)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        return emit.add("exp", _walk(node.left, names, emit), _walk(node.right, names, emit))
    raise ValueError("unsupported derivation")


def expression_program(derivation: str) -> str | None:
    """``(16,284 - 6,509) / 6,509`` -> ``subtract(16284, 6509), divide(#0, 6509)``."""
    names: dict[str, str] = {}

    def name(match: re.Match[str]) -> str:
        literal = match.group(0).replace(",", "").rstrip("%")
        key = f"n{len(names)}"
        names[key] = literal
        return key

    text = derivation.replace("$", "").replace("[", "(").replace("]", ")").replace("×", "*")
    text = _NUMBER_TOKEN.sub(name, text.replace("−", "-"))
    try:
        tree = ast.parse(text.strip(), mode="eval")
        emit = _Emit()
        result = _walk(tree.body, names, emit)
    except (SyntaxError, ValueError, RecursionError):
        return None
    if not emit.steps:
        return result if not result.startswith(("#", "const_")) else None
    return ", ".join(emit.steps)
