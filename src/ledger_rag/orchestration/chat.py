import json
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langsmith import tracing_context

from ledger_rag.core.config import Settings
from ledger_rag.core.contracts import Provider
from ledger_rag.core.models import Calculation, Citation, Principal, QueryRequest, Verification
from ledger_rag.generation.chat_tools import Retrieve, execute_tool
from ledger_rag.generation.evidence import validate_citations
from ledger_rag.generation.tools import CALCULATOR, SEARCH_EVIDENCE
from ledger_rag.generation.verification import AnswerVerifier
from ledger_rag.observability.spans import safe_span
from ledger_rag.orchestration.activity import ActivityCallback, activity
from ledger_rag.planning.planner import Planner, QueryPlan, recall

logger = structlog.get_logger()
# Defaults only (no validation side effects); used when no plan was recorded.
_DETERMINISTIC = Planner(Settings.model_construct())


class ChatState(TypedDict):
    messages: list[dict[str, Any]]
    citations: list[Citation]
    evidence_ids: set[str]
    tool_count: int
    pending_calls: list[dict[str, Any]]
    content: str
    calculations: NotRequired[list[Calculation]]
    tool_errors: NotRequired[int]
    verification: NotRequired[Verification | None]
    # Observability-only plan summary (no raw text); absent when planning is disabled.
    plan: NotRequired[dict[str, str | int | float | bool]]


class ChatUpdate(TypedDict, total=False):
    messages: list[dict[str, Any]]
    citations: list[Citation]
    evidence_ids: set[str]
    tool_count: int
    pending_calls: list[dict[str, Any]]
    content: str
    calculations: list[Calculation]
    tool_errors: int
    verification: Verification | None
    plan: dict[str, str | int | float | bool]


@dataclass(frozen=True)
class ChatContext:
    query: QueryRequest
    principal: Principal | None
    provider: Provider
    retrieve: Retrieve
    max_tool_calls: int
    on_activity: ActivityCallback | None
    verifier: AnswerVerifier | None = None
    # Optional explicit plan; otherwise the plan recorded by RagService.retrieve is reused.
    plan: QueryPlan | None = None
    planner: Planner | None = None


async def plan(state: ChatState, runtime: Runtime[ChatContext]) -> ChatUpdate:
    context = runtime.context
    found, recorded = recall(context.query)
    current = context.plan or recorded
    if current is None and found:
        return {}  # planner_enabled=false for this request
    async with activity(context.on_activity, "planning", span_name="rag.chat.plan") as span:
        if current is None:
            # Deterministic, offline plan: never re-runs the LLM rewrite or remote decisions.
            planner = context.planner or _DETERMINISTIC
            current = planner.deterministic(context.query, context.principal)
        summary = current.attributes()
        for key, value in summary.items():
            span.set_attribute(key, value)
    return {"plan": summary}


def after_plan(state: ChatState, runtime: Runtime[ChatContext]) -> Literal["model", "validation"]:
    # A confident out-of-scope refusal with no evidence skips the model and abstains.
    summary = state.get("plan") or {}
    if (
        summary.get("rag.plan.route") == "refuse_out_of_scope"
        and summary.get("rag.plan.subqueries") == 0
        and not state["citations"]
    ):
        return "validation"
    return entry(state, runtime)


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
    calculations = list(state.get("calculations", []))
    errors = state.get("tool_errors", 0)
    with safe_span("rag.chat.tools") as span:
        for call in state["pending_calls"]:
            message = await execute_tool(
                call,
                query=context.query,
                principal=context.principal,
                retrieve=context.retrieve,
                citations=citations,
                evidence_ids=evidence_ids,
                on_activity=context.on_activity,
                calculations=calculations,
            )
            errors += "error" in json.loads(message["content"])
            messages.append(message)
        count = state["tool_count"] + len(state["pending_calls"])
        span.set_attribute("rag.tool_calls", count)
    return {
        "messages": messages,
        "citations": citations,
        "evidence_ids": evidence_ids,
        "tool_count": count,
        "pending_calls": [],
        "calculations": calculations,
        "tool_errors": errors,
    }


async def validation(state: ChatState, runtime: Runtime[ChatContext]) -> ChatUpdate:
    context = runtime.context
    calculations = state.get("calculations", [])
    verification: Verification | None = None
    async with activity(context.on_activity, "validation", span_name="rag.chat.validation") as span:
        content, citations = validate_citations(state["content"], state["citations"])
        if context.verifier is not None:
            verification = context.verifier.check(content, citations, calculations)
            span.set_attribute("rag.verification", verification.status)
        span.set_attribute("rag.abstained", not bool(content))
        span.set_attribute("rag.citation_count", len(citations))
    if context.verifier is not None and verification is not None:
        verification = await context.verifier.verify_model(
            verification,
            content=content,
            question=context.query.question,
            citations=citations,
            profile=context.query.profile,
            on_activity=context.on_activity,
        )
    return {"content": content, "citations": citations, "verification": verification}


builder = StateGraph(ChatState, context_schema=ChatContext)
builder.add_node("plan", plan)
builder.add_node("model", model)
builder.add_node("tools", tools)
builder.add_node("validation", validation)
builder.add_edge(START, "plan")
builder.add_conditional_edges("plan", after_plan)
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
            config={"recursion_limit": 2 * context.max_tool_calls + 4, "callbacks": []},
        )
    return ChatState(
        messages=result["messages"],
        citations=result["citations"],
        evidence_ids=result["evidence_ids"],
        tool_count=result["tool_count"],
        pending_calls=result["pending_calls"],
        content=result["content"],
        calculations=result.get("calculations", []),
        tool_errors=result.get("tool_errors", 0),
        verification=result.get("verification"),
    )
