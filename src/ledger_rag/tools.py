import ast
from decimal import Decimal, InvalidOperation, localcontext

CALCULATOR = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Evaluate basic arithmetic from cited evidence. Use only numeric literals, "
            "+ - * / and parentheses. Returns decimal result; no external side effects."
        ),
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "maxLength": 200}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}

SEARCH_EVIDENCE = {
    "type": "function",
    "function": {
        "name": "search_evidence",
        "description": (
            "Retrieve additional authorized evidence when the original evidence is insufficient. "
            "Uses the original tenant, group permissions and as-of date. No web access."
        ),
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string", "minLength": 1, "maxLength": 4000}},
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}


def calculate(expression: str) -> str:
    if len(expression) > 200:
        raise ValueError("Expression exceeds 200 characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid arithmetic syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("Expression is too complex")

    def visit(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            source = ast.get_source_segment(expression, node)
            if source is None or len(source) > 40:
                raise ValueError("Invalid numeric literal")
            value = Decimal(source.replace("_", ""))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = visit(node.operand)
            value = -operand if isinstance(node.op, ast.USub) else operand
        elif isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            elif isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError("Division by zero")
                value = left / right
            else:
                raise ValueError("Unsupported arithmetic operator")
        else:
            raise ValueError("Only numeric arithmetic is allowed")
        if not value.is_finite() or abs(value) > Decimal("1e30"):
            raise ValueError("Result is outside the supported range")
        return value

    try:
        with localcontext() as context:
            context.prec = 28
            result = visit(tree.body)
    except (InvalidOperation, OverflowError) as exc:
        raise ValueError("Invalid arithmetic value") from exc
    return format(result, "f")
