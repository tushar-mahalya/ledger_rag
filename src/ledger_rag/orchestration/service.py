import time
from functools import partial

import structlog

from ledger_rag.core.config import Settings
from ledger_rag.core.contracts import Provider, Retriever
from ledger_rag.core.models import (
    Answer,
    Calculation,
    Citation,
    Hit,
    Principal,
    Profile,
    QueryRequest,
    Verification,
)
from ledger_rag.generation.abstention import AbstentionPolicy
from ledger_rag.generation.evidence import (
    ABSTENTION,
    citations_from_hits,
    generation_messages,
    validate_citations,
)
from ledger_rag.generation.verification import (
    AnswerVerifier,
    GroundednessVerifier,
    citation_coverage,
    make_groundedness_verifier,
)
from ledger_rag.observability.spans import safe_span
from ledger_rag.orchestration.activity import ActivityCallback, activity
from ledger_rag.orchestration.chat import ChatContext, ChatState, run_chat
from ledger_rag.retrieval.memory import authorized

logger = structlog.get_logger()


class RagService:
    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        provider: Provider,
        groundedness: GroundednessVerifier | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.provider = provider
        self.verifier = AnswerVerifier(
            settings, groundedness or make_groundedness_verifier(settings)
        )
        self.abstention = AbstentionPolicy.from_settings(settings)

    async def close(self) -> None:
        await self.verifier.close()

    async def retrieve(
        self,
        query: QueryRequest,
        principal: Principal,
        *,
        on_activity: ActivityCallback | None = None,
    ) -> tuple[list[Hit], float]:
        start = time.perf_counter()
        async with activity(on_activity, "retrieval") as span:
            effective = (
                query.model_copy(update={"top_k": min(query.top_k, 3)})
                if query.profile == Profile.VOICE
                else query
            )
            hits = await self.retriever.search(effective, principal)
            if any(not authorized(hit.chunk, principal, query) for hit in hits):
                logger.error("retrieval_authorization_violation")
                raise RuntimeError("Retriever returned unauthorized evidence")
            span.set_attribute("retrieval.hit_count", len(hits))
            span.set_attribute("rag.profile", query.profile.value)
        return hits, (time.perf_counter() - start) * 1000

    async def generate(
        self,
        query: QueryRequest,
        hits: list[Hit],
        retrieval_ms: float,
        request_id: str,
        principal: Principal | None = None,
        *,
        on_activity: ActivityCallback | None = None,
    ) -> Answer:
        if principal is not None and any(
            not authorized(hit.chunk, principal, query) for hit in hits
        ):
            raise RuntimeError("Unauthorized generation evidence")
        citations = citations_from_hits(hits)
        messages = generation_messages(query, citations)
        start = time.perf_counter()
        tool_count = tool_errors = 0
        calculations: list[Calculation] = []
        verification: Verification | None = None
        with safe_span("rag.generate") as span:
            span.set_attribute("gen_ai.request.model", self.provider.model)
            span.set_attribute("rag.profile", query.profile.value)
            if query.profile == Profile.CHAT:
                state = await run_chat(
                    ChatState(
                        messages=messages,
                        citations=citations,
                        evidence_ids={hit.chunk.id for hit in hits},
                        tool_count=0,
                        pending_calls=[],
                        content="",
                        calculations=[],
                        tool_errors=0,
                        verification=None,
                    ),
                    ChatContext(
                        query=query.model_copy(deep=True),
                        principal=principal.model_copy(deep=True) if principal else None,
                        provider=self.provider,
                        retrieve=partial(self.retrieve, on_activity=on_activity),
                        max_tool_calls=self.settings.max_tool_rounds,
                        on_activity=on_activity,
                        verifier=self.verifier,
                    ),
                )
                content, citations, tool_count = (
                    state["content"],
                    state["citations"],
                    state["tool_count"],
                )
                calculations, tool_errors = state["calculations"], state["tool_errors"]
                verification = state["verification"]
            else:
                content = ""
                if citations:
                    async with activity(on_activity, "generation"):
                        response = await self.provider.complete(messages, tools=None)
                        calls = response.get("tool_calls") or []
                        if not isinstance(calls, list):
                            raise ValueError("Provider returned invalid tool calls")
                        if not calls:
                            content = response.get("content") or ""
                            if not isinstance(content, str):
                                raise ValueError("Provider returned invalid content")
                async with activity(on_activity, "validation"):
                    content, citations = validate_citations(content, citations)
                    # Voice stays on the deterministic path: no model verifier, no extra calls.
                    verification = self.verifier.check(content, citations, [])
            decision = self.abstention.decide(
                content=content,
                hits=hits,
                verification=verification,
                citation_coverage=citation_coverage(content) if content else None,
                tool_calls=tool_count,
                tool_errors=tool_errors,
            )
            if decision.abstained:
                content, citations = "", []
            span.set_attribute("rag.tool_calls", tool_count)
            span.set_attribute("rag.abstained", decision.abstained)
            if verification is not None:
                span.set_attribute("rag.verification", verification.status)
        elapsed = (time.perf_counter() - start) * 1000
        generation_ms = elapsed if hits or (query.profile == Profile.CHAT and principal) else 0
        answer = self._answer(
            query, citations, content, retrieval_ms, generation_ms, request_id, tool_count
        )
        return answer.model_copy(
            update={
                "calculation": calculations[-1] if calculations else None,
                "calculations": calculations,
                "verification": verification,
                "abstention": decision,
            }
        )

    def _answer(
        self,
        query: QueryRequest,
        citations: list[Citation],
        content: str,
        retrieval_ms: float,
        generation_ms: float,
        request_id: str,
        tool_count: int,
    ) -> Answer:
        return Answer(
            answer=content or ABSTENTION,
            citations=citations,
            abstained=not bool(content),
            profile=query.profile,
            retrieval_ms=round(retrieval_ms, 3),
            generation_ms=round(generation_ms, 3),
            request_id=request_id,
            model=self.provider.model,
            tool_calls=tool_count,
        )
