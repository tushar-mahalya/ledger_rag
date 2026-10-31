from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langsmith import tracing_context

from ledger_rag.core.contracts import Provider
from ledger_rag.core.models import Citation, Principal, QueryRequest
from ledger_rag.generation.chat_tools import Retrieve, execute_tool
from ledger_rag.generation.evidence import validate_citations
from ledger_rag.generation.tools import CALCULATOR, SEARCH_EVIDENCE
from ledger_rag.observability.spans import safe_span
from ledger_rag.orchestration.activity import ActivityCallback, activity

logger = structlog.get_logger()


class ChatState(TypedDict):
    messages: list[dict[str, Any]]
    citations: list[Citation]
    evidence_ids: set[str]
    tool_count: int
    pending_calls: list[dict[str, Any]]
    content: str


class ChatUpdate(TypedDict, total=False):
    messages: list[dict[str, Any]]
    citations: list[Citation]
    evidence_ids: set[str]
    tool_count: int
    pending_calls: list[dict[str, Any]]
    content: str


@dataclass(frozen=True)
class ChatContext:
    query: QueryRequest
    principal: Principal | None
    provider: Provider
    retrieve: Retrieve
    max_tool_calls: int
    on_activity: ActivityCallback | None


def entry(state: ChatState, runtime: Runtime[ChatContext]) -> Literal["model", "validation"]:
    return "model" if state["citations"] or runtime.context.principal is not None else "validation"


async def model(state: ChatState, runtime: Runtime[ChatContext]) -> ChatUpdate:
    context = runtime.context
    tools = [CALCULATOR]
    if context.principal is not None:
        tools.append(SEARCH_EVIDENCE)
    async with activity(context.on_activity, "generation", span_name="rag.chat.model") as span:
        span.set_attribute("gen_ai.request.model", context.provider.model)
        response = await context.provider.complete(state["messages"], tools=tools)
        calls = response.get("tool_calls") or []
        if not isinstance(calls, list):
            raise ValueError("Provider returned invalid tool calls")
        if not calls:
            content = response.get("content") or ""
            if not isinstance(content, str):
                raise ValueError("Provider returned invalid content")
            return {"content": content, "pending_calls": []}
        if state["tool_count"] + len(calls) > context.max_tool_calls:
            logger.warning("tool_budget_exhausted", profile="chat")
            return {"content": "", "pending_calls": []}
        if any(not isinstance(call, dict) or not isinstance(call.get("id"), str) for call in calls):
            raise ValueError("Invalid tool call")
        return {
            "pending_calls": calls,
            "messages": [
                *state["messages"],
                {
                    "role": "assistant",
                    "content": response.get("content") or "",
                    "tool_calls": calls,
                },
            ],
        }


def after_model(state: ChatState) -> Literal["tools", "validation"]:
    return "tools" if state["pending_calls"] else "validation"


async def tools(state: ChatState, runtime: Runtime[ChatContext]) -> ChatUpdate:
    context = runtime.context
    messages, citations = list(state["messages"]), list(state["citations"])
    evidence_ids = set(state["evidence_ids"])
    with safe_span("rag.chat.tools") as span:
        for call in state["pending_calls"]:
            messages.append(
                await execute_tool(
                    call,
                    query=context.query,
                    principal=context.principal,
                    retrieve=context.retrieve,
                    citations=citations,
                    evidence_ids=evidence_ids,
                    on_activity=context.on_activity,
                )
            )
        count = state["tool_count"] + len(state["pending_calls"])
        span.set_attribute("rag.tool_calls", count)
    return {
        "messages": messages,
        "citations": citations,
        "evidence_ids": evidence_ids,
        "tool_count": count,
        "pending_calls": [],
    }


async def validation(state: ChatState, runtime: Runtime[ChatContext]) -> ChatUpdate:
    async with activity(
        runtime.context.on_activity, "validation", span_name="rag.chat.validation"
    ) as span:
        content, citations = validate_citations(state["content"], state["citations"])
        span.set_attribute("rag.abstained", not bool(content))
        span.set_attribute("rag.citation_count", len(citations))
        return {"content": content, "citations": citations}


builder = StateGraph(ChatState, context_schema=ChatContext)
builder.add_node("model", model)
builder.add_node("tools", tools)
builder.add_node("validation", validation)
builder.add_conditional_edges(START, entry)
builder.add_conditional_edges("model", after_model)
builder.add_edge("tools", "model")
builder.add_edge("validation", END)
CHAT_GRAPH = builder.compile(debug=False)


async def run_chat(state: ChatState, context: ChatContext) -> ChatState:
    # Never export graph state through environment-enabled LangSmith tracing.
    with tracing_context(enabled=False):
        result = await CHAT_GRAPH.ainvoke(
            state,
            context=context,
            config={"recursion_limit": 2 * context.max_tool_calls + 3, "callbacks": []},
        )
    return ChatState(
        messages=result["messages"],
        citations=result["citations"],
        evidence_ids=result["evidence_ids"],
        tool_count=result["tool_count"],
        pending_calls=result["pending_calls"],
        content=result["content"],
    )
