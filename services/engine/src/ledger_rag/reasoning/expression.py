"""Arithmetic expressions as the reader's program language, compiled to the FinQA DSL.

Small models reliably write ``(201 - 128) / 128 * 100`` but often misnumber FinQA's
step references (``divide(#0, 128), multiply(#0, const_100)``). Readers therefore
write ordinary expressions; this module compiles them, without ``eval``, to the
official step DSL that the deterministic executor runs:

- ``+ - * /`` and ``**`` map to ``add subtract multiply divide exp``;
- ``a > b`` maps to ``greater(a, b)`` (yes/no); ``a < b`` to ``greater(b, a)``;
- ``table_sum("row")`` (also ``table_average``, ``table_max``, ``table_min``)
  aggregates a table row;
- numbers may carry ``$``, thousands separators or a ``%`` sign (kept, as in
  FinQA: ``5%`` executes as 0.05); ``-x`` is a negative number.

:func:`render` does the reverse for gold programs, so training exemplars are shown in
the same language the reader must write. Both directions are checked over every
FinQA gold program by the test suite.
"""

import ast
import re

from ledger_rag.reasoning.dsl import TABLE_OPS, Program, ProgramParseError, candidate_text, parse

MAX_CHARS = 1000
_NUMBER = re.compile(r"(?<![\w.])\$?\s?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?<=\d)\s?%?")
_OPS = {ast.Add: "add", ast.Sub: "subtract", ast.Mult: "multiply", ast.Div: "divide"}
_BINARY = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "exp": "**"}


class _Compiler:
    def __init__(self, literals: dict[str, str]) -> None:
        self.literals = literals
        self.steps: list[str] = []

    def emit(self, op: str, left: str, right: str) -> str:
        self.steps.append(f"{op}({left}, {right})")
        return f"#{len(self.steps) - 1}"

    def node(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            if node.id in self.literals:
                return self.literals[node.id]
            if re.fullmatch(r"const_(?:m1|\d+(?:\.\d+)?)", node.id):
                return node.id
            raise ProgramParseError("unknown name in expression")
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return str(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            inner = self.node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return inner
            if inner.startswith(("#", "const_")):
                return self.emit("multiply", inner, "const_m1")
            return inner[1:] if inner.startswith("-") else "-" + inner
        if isinstance(node, ast.BinOp):
            power = isinstance(node.op, ast.Pow | ast.BitXor)  # "^" means a power here.
            op = _OPS.get(type(node.op)) or ("exp" if power else None)
            if op is None:
                raise ProgramParseError("unsupported operator in expression")
            return self.emit(op, self.node(node.left), self.node(node.right))
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = self.node(node.left), self.node(node.comparators[0])
            if isinstance(node.ops[0], ast.Gt):
                return self.emit("greater", left, right)
            if isinstance(node.ops[0], ast.Lt):
                return self.emit("greater", right, left)
            raise ProgramParseError("only > and < comparisons are supported")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in TABLE_OPS or len(node.args) != 1 or node.keywords:
                raise ProgramParseError("unknown function in expression")
            row = node.args[0]
            if not isinstance(row, ast.Constant) or not isinstance(row.value, str):
                raise ProgramParseError("a table function takes a quoted row name")
            self.steps.append(f"{node.func.id}({row.value.strip()}, none)")
            return f"#{len(self.steps) - 1}"
        raise ProgramParseError("unsupported expression")


def compile_expression(text: str) -> Program:
    """``(201 - 128) / 128 * 100`` -> the equivalent FinQA DSL :class:`Program`."""
    source = text.strip().strip("`").strip()
    if not source or len(source) > MAX_CHARS:
        raise ProgramParseError("empty or overlong expression")
    literals: dict[str, str] = {}
    strings: list[str] = []

    def keep_string(match: re.Match[str]) -> str:
        strings.append(match.group(0))
        return f"__s{len(strings) - 1}__"

    # Protect quoted row names, then replace each number with a placeholder name.
    masked = re.sub(r"\"[^\"]*\"|'[^']*'", keep_string, source)

    def literal(match: re.Match[str]) -> str:
        raw = match.group(0).replace("$", "").replace(",", "").replace(" ", "")
        if not raw or raw == "%":
            return match.group(0)
        name = f"n{len(literals)}"
        literals[name] = raw
        return name

    masked = _NUMBER.sub(literal, masked).replace("×", "*").replace("−", "-")
    for index, value in enumerate(strings):
        masked = masked.replace(f"__s{index}__", value)
    try:
        tree = ast.parse(masked, mode="eval")
    except SyntaxError:
        raise ProgramParseError("not a valid expression") from None
    compiler = _Compiler(literals)
    result = compiler.node(tree.body)
    if not compiler.steps:
        if result.startswith("const_"):  # A bare constant is the number it names.
            return Program((), "-1" if result == "const_m1" else result.removeprefix("const_"))
        if result.startswith("#"):
            raise ProgramParseError("an expression needs a number")
        return Program((), result)
    return parse(", ".join(compiler.steps))


def read_program(text: str) -> Program:
    """The program of a reply: an expression after ``PROGRAM:``, or FinQA DSL calls."""
    body = candidate_text(text)
    if is_dsl(body) or not body.strip():
        return parse(text)
    line = next((ln for ln in body.splitlines() if ln.strip()), "")
    line = line.strip().strip("`*").strip().rstrip(".")
    line = re.split(r"\s*(?:=|≈)\s*", line, maxsplit=1)[0]  # "(a - b) / b = 0.57"
    return compile_expression(line)


def render(program: Program) -> str:
    """Gold DSL -> the expression a reader would write (the answer is the last step)."""
    if program.value is not None:
        return program.value
    parts: list[str] = []
    for step in program.steps:
        args = [parts[int(a[1:])] if a.startswith("#") else a for a in step.args]
        if step.op in TABLE_OPS:
            parts.append(f'{step.op}("{step.args[0]}")')
        elif step.op == "greater":
            parts.append(f"({args[0]} > {args[1]})")
        else:
            parts.append(f"({args[0]} {_BINARY[step.op]} {args[1]})")
    expression = parts[-1]
    return (
        expression[1:-1] if expression.startswith("(") and expression.endswith(")") else expression
    )


def is_dsl(text: str) -> bool:
    """True when ``text`` already uses FinQA DSL calls rather than an expression."""
    return bool(re.search(r"\b(?:add|subtract|multiply|divide|exp|greater)\s*\(", text))
