"""Parse model-written FinQA DSL programs into a bounded, canonical form.

Grammar (FinQA): ``op(arg, arg), op(#0, arg), ...`` where ``op`` is one of
``add subtract multiply divide exp greater`` (two operands) or ``table_max table_min
table_sum table_average`` (``row name, none``). An operand is a number (``5829``,
``3.5%``), a constant (``const_100``, ``const_m1``) or ``#k``, the result of step
``k``. The final step is the answer. A ConvFinQA lookup turn may instead be a single
bare number (``PROGRAM: 60.94``): the answer is that number as read from the evidence.

Models wrap programs in prose, code fences or quotes and write ``$5,829``; this
module recovers exactly one program or raises :class:`ProgramParseError`. The
canonical text is what the official tokenizer and executor consume.
"""

import re
from dataclasses import dataclass

BINARY_OPS = frozenset({"add", "subtract", "multiply", "divide", "exp", "greater"})
TABLE_OPS = frozenset({"table_max", "table_min", "table_sum", "table_average"})
OPS = BINARY_OPS | TABLE_OPS
MAX_STEPS = 10
MAX_PROGRAM_CHARS = 2000

_STEP = re.compile(
    r"\b(?P<op>" + "|".join(sorted(OPS, key=len, reverse=True)) + r")\s*\((?P<args>[^()]*)\)"
)
_REF = re.compile(r"#(\d+)")
_CONST = re.compile(r"const_(?:m1|\d+(?:\.\d+)?)")
_NUMBER = re.compile(r"-?(?:\d+(?:\.\d+)?|\.\d+)%?")
_MARKER = re.compile(r"^\s*(?:\*\*)?program(?:\*\*)?\s*[:=]\s*", re.IGNORECASE | re.MULTILINE)


class ProgramParseError(ValueError):
    """The text holds no single well-formed program; messages never echo model text."""


@dataclass(frozen=True)
class Step:
    op: str
    args: tuple[str, str]

    def render(self) -> str:
        return f"{self.op}({self.args[0]}, {self.args[1]})"


@dataclass(frozen=True)
class Program:
    steps: tuple[Step, ...]
    value: str | None = None
    """A bare-number program (lookup turn); set only when ``steps`` is empty."""

    @property
    def text(self) -> str:
        """Canonical form accepted by the official FinQA tokenizer (or the bare number)."""
        if self.value is not None:
            return self.value
        return ", ".join(step.render() for step in self.steps)

    def literals(self) -> list[str]:
        """Numeric operands (not constants, step references or table rows)."""
        if self.value is not None:
            return [self.value]
        return [
            arg
            for step in self.steps
            if step.op in BINARY_OPS
            for arg in step.args
            if not arg.startswith("#") and not arg.startswith("const_")
        ]

    def table_rows(self) -> list[str]:
        return [step.args[0] for step in self.steps if step.op in TABLE_OPS]


def _number(raw: str) -> str:
    """Normalize ``$5,829.0``, ``(1,234)`` or ``5.2 %`` to an official numeric token."""
    text = raw.strip().strip("'\"`")
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace("$", "").replace(",", "").replace(" ", "")
    if text.startswith("+"):
        text = text[1:]
    if not _NUMBER.fullmatch(text):
        raise ProgramParseError("operand is not a number, constant or step reference")
    if negative and not text.startswith("-"):
        text = "-" + text
    return text


def _operand(raw: str, step: int) -> str:
    text = raw.strip().strip("'\"`")
    ref = _REF.fullmatch(text)
    if ref:
        if int(ref.group(1)) >= step:
            raise ProgramParseError("a step may only reference earlier steps")
        return text
    if _CONST.fullmatch(text):
        return text
    return _number(text)


def _split_binary(args: str) -> tuple[str, str]:
    """Split two operands, preferring ``, `` so ``5,829`` stays one number."""
    for separator in (r",\s+", r"\s*,\s*"):
        parts = re.split(separator, args.strip())
        if len(parts) == 2 and all(part.strip() for part in parts):
            return parts[0], parts[1]
    raise ProgramParseError("a binary operation needs exactly two operands")


def _step(op: str, args: str, index: int) -> Step:
    if op in TABLE_OPS:
        # The official executor reads only the row name; the second operand is
        # conventionally ``none`` and kept verbatim so gold programs round-trip.
        row, _, tail = args.rpartition(",")
        name, second = row.strip().strip("'\"`"), tail.strip().strip("'\"`")
        if not name or not second:
            raise ProgramParseError("a table operation takes (row name, none)")
        if _REF.fullmatch(name):
            raise ProgramParseError("a table operation takes a row name, not a step")
        return Step(op, (name, second))
    first, second = _split_binary(args)
    return Step(op, (_operand(first, index), _operand(second, index)))


def _candidate(text: str) -> str:
    """The paragraph after the last ``PROGRAM:`` marker, else the last one with a step."""
    markers = list(_MARKER.finditer(text))
    if markers:
        return text[markers[-1].end() :].lstrip().split("\n\n")[0]
    paragraphs = [part for part in re.split(r"\n\s*\n", text) if _STEP.search(part)]
    return paragraphs[-1] if paragraphs else ""


def parse(text: str) -> Program:
    """Recover the program from a model reply; raises :class:`ProgramParseError`."""
    if len(text) > 20 * MAX_PROGRAM_CHARS:
        raise ProgramParseError("reply is too long")
    body = _candidate(text)
    matches = list(_STEP.finditer(body))
    if not matches:
        bare = (body if _MARKER.search(text) else text).strip().strip("`*").strip()
        try:
            return Program((), _number(bare.splitlines()[0] if bare else ""))
        except ProgramParseError:
            raise ProgramParseError("no program found") from None
    if len(matches) > MAX_STEPS:
        raise ProgramParseError("program has too many steps")
    steps = tuple(_step(m.group("op"), m.group("args"), i) for i, m in enumerate(matches))
    program = Program(steps)
    if len(program.text) > MAX_PROGRAM_CHARS:
        raise ProgramParseError("program is too long")
    return program
