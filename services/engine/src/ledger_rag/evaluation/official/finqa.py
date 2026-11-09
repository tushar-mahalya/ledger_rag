"""FinQA / ConvFinQA execution accuracy with the official executor.

Two comparisons are reported, never mixed:

- ``official``: the upstream ``eval_program`` result (rounded to 5 decimals) must
  equal the gold ``exe_ans`` exactly, as in the official ``evaluate.py``. This is the
  metric of FinQANet, APOLLO, ENCORE and the CodaLab leaderboard.
- ``tolerant``: Program-of-Thoughts' ``finqa_equal`` without percentage scaling:
  both values rounded to the smaller of their decimal precisions must be equal
  (https://github.com/TIGER-AI-Lab/Program-of-Thoughts, ``tool.py``, MIT). LLM
  prompting papers (PoT, FINDER) report this looser number.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ledger_rag.evaluation.official import _finqa_upstream

# Untyped upstream code is called through ``Any`` so it stays byte-identical.
_UPSTREAM: Any = _finqa_upstream
OPS: tuple[str, ...] = tuple(_UPSTREAM.all_ops)
Answer = float | str


@dataclass(frozen=True)
class Execution:
    """Result of running a program with the official executor."""

    value: Answer | None
    valid: bool


def tokens(program: str) -> list[str]:
    """Official tokenization of a ``op(a, b), op(#0, c)`` program string."""
    return list(_UPSTREAM.program_tokenization(program))


def execute(program: str, table: Sequence[Sequence[str]]) -> Execution:
    """Run ``program`` over the FinQA ``table`` exactly as the official evaluator does.

    ConvFinQA lookup turns are a bare number (``"60.94"``); the official FinQA
    executor cannot run those, so their value is the upstream ``str_to_num`` rounded
    to 5 places, which reproduces every ConvFinQA ``exe_ans_list`` entry.
    """
    if "(" not in program:
        number = _UPSTREAM.str_to_num(program.strip())
        if number == "n/a":
            return Execution(None, False)
        return Execution(round(float(number), 5), True)
    invalid, value = _UPSTREAM.eval_program(tokens(program), [list(row) for row in table])
    if invalid or value == "n/a":
        return Execution(None, False)
    if isinstance(value, str):
        return Execution(value, True)
    if isinstance(value, complex) or not math.isfinite(float(value)):
        return Execution(None, False)
    return Execution(float(value), True)


def official_correct(value: Answer | None, gold: Answer) -> bool:
    """Official execution accuracy: exact equality with the gold ``exe_ans``."""
    return value is not None and value == gold


def _precision(value: float) -> int:
    text = str(value)
    return len(text.split(".")[-1]) if "." in text else 5


def tolerant_correct(value: Answer | None, gold: Answer) -> bool:
    """PoT ``finqa_equal(prediction, reference, include_percentage=False)``."""
    if value is None:
        return False
    if isinstance(value, str) or isinstance(gold, str):
        return value == gold
    places = min(_precision(value), _precision(gold))
    return round(value, places) == round(gold, places)
