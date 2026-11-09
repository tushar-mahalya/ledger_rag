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

Unambiguous slips are repaired deterministically, never guessed: nested calls
(``divide(add(a, b), 2)``) and n-ary ``add``/``multiply`` become sequential binary
steps, and consistently 1-based references (``#1`` meaning the first step, with no
``#0`` anywhere) are shifted to FinQA's 0-based convention.
"""

import re
from dataclasses import dataclass

BINARY_OPS = frozenset({"add", "subtract", "multiply", "divide", "exp", "greater"})
TABLE_OPS = frozenset({"table_max", "table_min", "table_sum", "table_average"})
OPS = BINARY_OPS | TABLE_OPS
MAX_STEPS = 16
MAX_PROGRAM_CHARS = 2000

_STEP = re.compile(
    r"\b(?P<op>" + "|".join(sorted(OPS, key=len, reverse=True)) + r")\s*\((?P<args>[^()]*)\)"
)
_REF = re.compile(r"#(\d+)")
_CONST = re.compile(r"const_(?:m1|\d+(?:\.\d+)?)")
_NUMBER = re.compile(r"-?(?:\d+(?:\.\d+)?|\.\d+)%?")
_MARKER = re.compile(r"^\s*(?:\*\*)?program(?:\*\*)?\s*[:=]\s*", re.IGNORECASE | re.MULTILINE)


def _constant_value(constant: str) -> str:
    """``const_100`` -> ``100``; ``const_m1`` -> ``-1`` (constants are checked too)."""
    name = constant.removeprefix("const_")
    return "-1" if name == "m1" else name


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
        """Numeric operands, constants as the numbers they name (not steps or rows)."""
        if self.value is not None:
            return [self.value]
        return [
            _constant_value(arg) if arg.startswith("const_") else arg
            for step in self.steps
            if step.op in BINARY_OPS
            for arg in step.args
            if not arg.startswith("#")
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


@dataclass
class _Call:
    op: str
    args: list["_Call | str"]


_OP_AT = re.compile(r"(?P<op>" + "|".join(sorted(OPS, key=len, reverse=True)) + r")\s*\(")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


class _Scanner:
    """Recursive-descent reader for ``op(arg, ...)`` calls, tolerant of nesting."""

    def __init__(self, text: str) -> None:
        self.text = _THOUSANDS.sub("", text)
        self.pos = 0

    def calls(self) -> list[_Call]:
        found: list[_Call] = []
        while (match := _OP_AT.search(self.text, self.pos)) is not None:
            before = self.text[match.start() - 1] if match.start() else " "
            if before.isalnum() or before == "_":
                self.pos = match.end()
                continue
            self.pos = match.start()
            found.append(self.call())
            if len(found) > MAX_STEPS:
                raise ProgramParseError("program has too many steps")
        return found

    def call(self) -> _Call:
        match = _OP_AT.match(self.text, self.pos)
        if match is None:
            raise ProgramParseError("expected an operation")
        op = match.group("op")
        self.pos = match.end()
        if op in TABLE_OPS:
            return _Call(op, [self._balanced()])
        args: list[_Call | str] = []
        while True:
            self._skip_space()
            if _OP_AT.match(self.text, self.pos):
                args.append(self.call())
            else:
                args.append(self._atom())
            self._skip_space()
            if self.pos >= len(self.text):
                raise ProgramParseError("unbalanced parentheses")
            char = self.text[self.pos]
            self.pos += 1
            if char == ")":
                return _Call(op, args)
            if char != ",":
                raise ProgramParseError("operands must be separated by commas")

    def _skip_space(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _atom(self) -> str:
        depth, start = 0, self.pos
        while self.pos < len(self.text):
            char = self.text[self.pos]
            if char == "(":
                depth += 1
            elif char == ")":
                if depth == 0:
                    break
                depth -= 1
            elif char == "," and depth == 0:
                break
            self.pos += 1
        return self.text[start : self.pos].strip()

    def _balanced(self) -> str:
        depth, start = 0, self.pos
        while self.pos < len(self.text):
            char = self.text[self.pos]
            if char == "(":
                depth += 1
            elif char == ")":
                if depth == 0:
                    self.pos += 1
                    return self.text[start : self.pos - 1]
                depth -= 1
            self.pos += 1
        raise ProgramParseError("unbalanced parentheses")


def _refs(call: _Call) -> list[int]:
    out: list[int] = []
    for arg in call.args:
        if isinstance(arg, _Call):
            out.extend(_refs(arg))
        elif (ref := _REF.fullmatch(arg.strip().strip("'\"`"))) is not None:
            out.append(int(ref.group(1)))
    return out


def _one_based(calls: list[_Call]) -> bool:
    """True when step references are consistently 1-based (``#1`` = the first step)."""
    refs = [(index, ref) for index, call in enumerate(calls) for ref in _refs(call)]
    if not refs or any(ref == 0 for _, ref in refs):
        return False
    return any(ref >= index for index, ref in refs) and all(ref <= index for index, ref in refs)


class _Flattener:
    """Emit nested and n-ary calls as sequential binary steps; remap references."""

    def __init__(self, shift: int) -> None:
        self.steps: list[Step] = []
        self.top: dict[int, int] = {}
        self.shift = shift

    def emit(self, call: _Call, top_index: int) -> int:
        if call.op in TABLE_OPS:
            row, _, tail = str(call.args[0]).rpartition(",")
            name, second = row.strip().strip("'\"`"), tail.strip().strip("'\"`")
            if not name or not second:
                raise ProgramParseError("a table operation takes (row name, none)")
            if _REF.fullmatch(name):
                raise ProgramParseError("a table operation takes a row name, not a step")
            return self._append(Step(call.op, (name, second)))
        operands = [
            f"#{self.emit(arg, top_index)}"
            if isinstance(arg, _Call)
            else self._atom(arg, top_index)
            for arg in call.args
        ]
        if len(operands) == 2:
            return self._append(Step(call.op, (operands[0], operands[1])))
        if len(operands) > 2 and call.op in {"add", "multiply"}:
            index = self._append(Step(call.op, (operands[0], operands[1])))
            for operand in operands[2:]:
                index = self._append(Step(call.op, (f"#{index}", operand)))
            return index
        raise ProgramParseError("a binary operation needs exactly two operands")

    def _atom(self, raw: str, top_index: int) -> str:
        text = raw.strip().strip("'\"`")
        ref = _REF.fullmatch(text)
        if ref:
            target = int(ref.group(1)) - self.shift
            if not 0 <= target < top_index:
                raise ProgramParseError("a step may only reference earlier steps")
            return f"#{self.top[target]}"
        if _CONST.fullmatch(text):
            return text
        return _number(text)

    def _append(self, step: Step) -> int:
        self.steps.append(step)
        if len(self.steps) > MAX_STEPS:
            raise ProgramParseError("program has too many steps")
        return len(self.steps) - 1


def candidate_text(text: str) -> str:
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
    body = candidate_text(text)
    calls = _Scanner(body).calls()
    if not calls:
        bare = (body if _MARKER.search(text) else text).strip().strip("`*").strip()
        try:
            return Program((), _number(bare.splitlines()[0] if bare else ""))
        except ProgramParseError:
            raise ProgramParseError("no program found") from None
    flattener = _Flattener(shift=1 if _one_based(calls) else 0)
    for index, call in enumerate(calls):
        flattener.top[index] = flattener.emit(call, index)
    program = Program(tuple(flattener.steps))
    if len(program.text) > MAX_PROGRAM_CHARS:
        raise ProgramParseError("program is too long")
    return program
