from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from opentelemetry.trace import Span

from ledger_rag.core.models import (
    ACTIVITY_KINDS,
    Activity,
    ActivityKind,
    ActivityStage,
    ActivityTool,
    PlanSummary,
)
from ledger_rag.observability.spans import safe_span

ActivityCallback = Callable[[Activity], Awaitable[None]]

_STAGE_KINDS: dict[ActivityStage, ActivityKind] = {
    "planning": "planning",
    "retrieval": "retrieval",
    "generation": "generation",
    "validation": "citation_validation",
}


def activity_kind(stage: ActivityStage, tool: ActivityTool | None = None) -> ActivityKind:
    if stage != "tool":
        return _STAGE_KINDS[stage]
    if tool == "calculator":
        return "calculator"
    if tool == "search_evidence":
        return "evidence_search"
    return "tool_check"


@asynccontextmanager
async def activity(
    callback: ActivityCallback | None,
    stage: ActivityStage,
    *,
    tool: ActivityTool | None = None,
    span_name: str | None = None,
    kind: ActivityKind | None = None,
    plan: PlanSummary | None = None,
) -> AsyncIterator[Span]:
    """Emit started/completed activities around a span; ``plan`` rides on completion."""
    kind = kind or activity_kind(stage, tool)
    expected_stage, label = ACTIVITY_KINDS[kind]
    if expected_stage != stage:
        raise ValueError("Activity kind does not belong to this stage")
    with safe_span(span_name or f"rag.{stage}") as span:
        span.set_attribute("rag.stage", stage)
        span.set_attribute("rag.activity", kind)
        if tool is not None:
            span.set_attribute("rag.tool", tool)
        if callback is not None:
            await callback(
                Activity(stage=stage, status="started", label=label, tool=tool, kind=kind)
            )
        yield span
        if callback is not None:
            await callback(
                Activity(
                    stage=stage, status="completed", label=label, tool=tool, kind=kind, plan=plan
                )
            )
