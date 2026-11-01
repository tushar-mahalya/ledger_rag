import json
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from ledger_rag.core.models import (
    MAX_CITATIONS,
    ActivityTool,
    Calculation,
    Citation,
    Hit,
    Principal,
    QueryRequest,
)
from ledger_rag.generation.evidence import citation_from_hit, evidence_payload
from ledger_rag.generation.program import ProgramError, program_tool_result, run_program
from ledger_rag.orchestration.activity import ActivityCallback, activity

Retrieve = Callable[[QueryRequest, Principal], Awaitable[tuple[list[Hit], float]]]
logger = structlog.get_logger()


async def execute_tool(
    call: dict[str, Any],
    *,
    query: QueryRequest,
    principal: Principal | None,
    retrieve: Retrieve,
    citations: list[Citation],
    evidence_ids: set[str],
    on_activity: ActivityCallback | None,
    calculations: list[Calculation] | None = None,
) -> dict[str, Any]:
    name, arguments = call.get("name"), call.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    allowed_search = (
        name == "search_evidence"
        and principal is not None
        and set(arguments) == {"question"}
        and isinstance(arguments["question"], str)
        and 0 < len(arguments["question"].strip()) <= 4000
    )
    # Program arguments are validated by run_program, which returns correctable errors.
    allowed_calculator = name == "calculator" and isinstance(call.get("arguments"), dict)
    tool: ActivityTool | None = (
        "search_evidence" if allowed_search else "calculator" if allowed_calculator else None
    )
    async with activity(on_activity, "tool", tool=tool):
        result: dict[str, Any]
        if allowed_search and principal is not None:
            # Only the question changes; all filters and the original principal survive.
            search_query = query.model_copy(update={"question": arguments["question"]})
            additional, _ = await retrieve(search_query, principal)
            added = []
            for hit in additional:
                if hit.chunk.id in evidence_ids or len(citations) >= MAX_CITATIONS:
                    continue
                evidence_ids.add(hit.chunk.id)
                citation = citation_from_hit(hit, len(citations) + 1)
                citations.append(citation)
                added.append(evidence_payload(citation))
            result = {"evidence": added, "exhausted": not bool(added)}
        elif allowed_calculator:
            try:
                calculation = run_program(arguments, citations)
            except ProgramError as exc:
                result = {"error": str(exc)}
                logger.warning("tool_input_rejected")
            except (TypeError, ValueError, ArithmeticError, KeyError, IndexError) as exc:
                # Defense in depth: malformed arguments must never become a 500.
                result = {"error": "calculator arguments could not be parsed; check the schema"}
                logger.warning("tool_input_rejected", error_class=type(exc).__name__)
            else:
                if calculations is not None:
                    calculations.append(calculation)
                result = program_tool_result(calculation)
        else:
            result = {"error": "Tool or arguments are not allowed"}
            logger.warning("tool_call_rejected")
    return {
        "role": "tool",
        "tool_call_id": call["id"],
        "name": name,
        "content": json.dumps(result),
    }
