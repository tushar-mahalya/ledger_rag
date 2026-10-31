from ledger_rag.generation.program import MAX_ARGS, MAX_QUOTE_CHARS, MAX_STEPS, OPS

_OPERAND = {
    "type": "object",
    "description": (
        "Exactly one of: {citation_id, quote} where quote is an exact span of that evidence "
        "text containing one number; {ref: '#n'} for the result of earlier step n (0-based); "
        "or {constant} limited to 1, 2, 4, 12, 100, 1000 or 1000000."
    ),
    "properties": {
        "citation_id": {"type": "string", "maxLength": 8},
        "quote": {"type": "string", "minLength": 1, "maxLength": MAX_QUOTE_CHARS},
        "ref": {"type": "string", "maxLength": 3},
        "constant": {"type": "number"},
    },
    "additionalProperties": False,
}

CALCULATOR = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Execute a short FinQA-style program over cited evidence with exact decimal "
            "arithmetic. Numbers must be bound to evidence by quoting the exact span; free "
            "numeric literals are rejected. Scale words (thousand/million/billion) are applied "
            "and percentages become fractions. percent_change(old, new) returns a percentage; "
            "greater returns 1 or 0. Returns the result and a step trace with provenance."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_STEPS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "enum": list(OPS)},
                            "args": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": MAX_ARGS,
                                "items": _OPERAND,
                            },
                        },
                        "required": ["op", "args"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["steps"],
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
