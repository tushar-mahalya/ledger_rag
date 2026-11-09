"""Check that every number a program uses was read from the evidence or the question.

A literal is grounded when its magnitude equals a number mentioned in the evidence
or question, either as written (``5,829`` in ``$5,829 million``), in base units
(``5829000000``) or a percentage as a fraction (``0.05`` for ``5%``), ignoring sign
(filings show decreases as ``(1,234)`` or in prose).
A trailing ``%`` on the literal is notation only; the digits must still match.
Constants (``const_100``, small counts and unit factors in :data:`CONSTANTS`) and
step references are always grounded; table rows must name a row of the evidence
table (:func:`repair_rows` first maps case or spelling slips to the unique close row).
Ungrounded programs are treated as invalid samples: they cannot win a vote, so a
reader cannot invent its own inputs.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from difflib import get_close_matches

from ledger_rag.generation.program import extract_numbers
from ledger_rag.reasoning.dsl import TABLE_OPS, Program, Step

# Counts and unit factors a program may use without quoting them (FinQA's constants:
# "average over 3 years", "in thousands"). Any other number must come from evidence.
CONSTANTS = frozenset(Decimal(v) for v in (*range(1, 13), 100, 1000, 10000, 1000000))

# Spelled-out counts in the evidence or question that annotators use ("over three years").
_WORDS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twelve": 12,
    "half": 2,
    "quarter": 4,
    "double": 2,
    "twice": 2,
    "triple": 3,
}


def _key(value: Decimal) -> Decimal:
    return abs(value).normalize()


@dataclass(frozen=True)
class Evidence:
    """Magnitudes stated in the evidence/question and the table's row names."""

    numbers: frozenset[Decimal]
    rows: frozenset[str]

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        table: Sequence[Sequence[str]] = (),
        question: str = "",
    ) -> "Evidence":
        numbers: set[Decimal] = set(CONSTANTS)
        cells = [cell for row in table for cell in row]
        sources = [*texts, *cells, question]
        for text in sources:
            for mention in extract_numbers(text):
                numbers.add(_key(mention.value))
                numbers.add(_key(mention.absolute))
                numbers.add(_key(mention.operand_value))  # "5%" may be written 0.05.
        lowered = " ".join(sources).lower()
        numbers.update(
            Decimal(value) for word, value in _WORDS.items() if re.search(rf"\b{word}\b", lowered)
        )
        rows = frozenset(row[0] for row in table if row)
        return cls(frozenset(numbers), rows)


def _literal_value(literal: str) -> Decimal | None:
    try:
        return _key(Decimal(literal.rstrip("%")))
    except InvalidOperation:
        return None


def ungrounded(program: Program, evidence: Evidence) -> list[str]:
    """Literals and table rows in ``program`` that the evidence does not support."""
    missing = [
        literal
        for literal in program.literals()
        if (value := _literal_value(literal)) is None or value not in evidence.numbers
    ]
    missing.extend(row for row in program.table_rows() if row not in evidence.rows)
    return missing


def repair_rows(program: Program, evidence: Evidence) -> Program:
    """Map a table row a reader misspelled or re-cased to its unique close match."""
    rows = sorted(evidence.rows)
    lowered = {row.lower(): row for row in rows}
    steps: list[Step] = []
    for step in program.steps:
        name = step.args[0]
        if step.op in TABLE_OPS and name not in evidence.rows:
            match = lowered.get(name.lower()) or next(
                iter(get_close_matches(name, rows, n=1, cutoff=0.85)), None
            )
            if match is not None:
                step = Step(step.op, (match, step.args[1]))
        steps.append(step)
    return replace(program, steps=tuple(steps))
