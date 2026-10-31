from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from opentelemetry.trace import Span

from ledger_rag.core.models import Activity, ActivityLabel, ActivityStage, ActivityTool
from ledger_rag.observability.spans import safe_span

ActivityCallback = Callable[[Activity], Awaitable[None]]


@asynccontextmanager
async def activity(
    callback: ActivityCallback | None,
    stage: ActivityStage,
    *,
    tool: ActivityTool | None = None,
    span_name: str | None = None,
) -> AsyncIterator[Span]:
    label: ActivityLabel
    if stage == "tool":
        if tool == "calculator":
            label = "Calculating"
        elif tool == "search_evidence":
            label = "Searching evidence"
        else:
            label = "Checking tool request"
    else:
        labels: dict[ActivityStage, ActivityLabel] = {
            "planning": "Planning query",
            "retrieval": "Retrieving evidence",
            "generation": "Generating answer",
            "validation": "Validating citations",
        }
        label = labels[stage]
    with safe_span(span_name or f"rag.{stage}") as span:
        span.set_attribute("rag.stage", stage)
        if tool is not None:
            span.set_attribute("rag.tool", tool)
        if callback is not None:
            await callback(Activity(stage=stage, status="started", label=label, tool=tool))
        yield span
        if callback is not None:
            await callback(Activity(stage=stage, status="completed", label=label, tool=tool))
