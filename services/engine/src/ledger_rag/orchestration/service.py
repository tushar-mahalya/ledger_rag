import asyncio
import time
from functools import partial

import structlog
from opentelemetry.trace import Span

from ledger_rag.core.config import Settings
from ledger_rag.core.contracts import Provider, Retriever
from ledger_rag.core.models import (
    MAX_CALCULATIONS,
    Answer,
    Calculation,
    Hit,
    Principal,
    Profile,
    QueryRequest,
    Verification,
)
from ledger_rag.external.service import ExternalData, ExternalSession
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
from ledger_rag.planning.decomposition import reciprocal_rank_fusion
from ledger_rag.planning.planner import Planner, make_planner, recall, remember
from ledger_rag.retrieval.memory import authorized

logger = structlog.get_logger()


class RagService:
    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        provider: Provider,
        *,
        groundedness: GroundednessVerifier | None = None,
        planner: Planner | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.provider = provider
        self.verifier = AnswerVerifier(
            settings, groundedness or make_groundedness_verifier(settings)
        )
        self.abstention = AbstentionPolicy.from_settings(settings)
        self.planner = planner or make_planner(settings, provider)
        self.external = ExternalData.from_settings(settings)

    async def close(self) -> None:
        """Release clients the service owns: the planner and groundedness verifier.

        The retriever and provider are injected and closed by their owner.
        """
        try:
            await self.planner.close()
        finally:
            try:
                await self.verifier.close()
            finally:
                if self.external is not None:
                    await self.external.aclose()

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
            if self.settings.planner_enabled:
                hits = await self._planned_search(effective, principal, span)
            else:
                remember(query, None)
                hits = await self.retriever.search(effective, principal)
            if any(not authorized(hit.chunk, principal, query) for hit in hits):
                logger.error("retrieval_authorization_violation")
                raise RuntimeError("Retriever returned unauthorized evidence")
            span.set_attribute("retrieval.hit_count", len(hits))
            span.set_attribute("rag.profile", query.profile.value)
        return hits, (time.perf_counter() - start) * 1000

    async def _planned_search(
        self, query: QueryRequest, principal: Principal, span: Span
    ) -> list[Hit]:
        """Rewrite/decompose, search sub-queries concurrently and fuse them with RRF.

        Only ``question`` (and a bounded ``top_k`` overfetch) differs between sub-queries;
        profile, ``as_of`` and the principal-derived filters are those of the request.
        """
        with safe_span("rag.plan") as plan_span:
            # Voice gets no LLM rewrite/decomposition; only the decision engine may be remote.
            plan = await self.planner.plan(query, principal)
            for key, value in plan.attributes().items():
                plan_span.set_attribute(key, value)
        remember(query, plan)
        span.set_attribute("rag.plan.subqueries", len(plan.subqueries))
        if plan.skip_retrieval:
            return []
        constrained = bool(plan.analysis.companies)
        if plan.subqueries == (query.question,) and not constrained:
            return await self.retriever.search(query, principal)
        # Overfetch only for chat; the voice budget keeps its top_k.
        fetch = query.top_k
        if constrained and query.profile == Profile.CHAT:
            fetch = min(10, 2 * query.top_k)
        searcher = self.planner.narrow(plan, principal)
        limit = asyncio.Semaphore(self.settings.planner_max_concurrency)

        async def search(question: str) -> list[Hit]:
            async with limit:
                sub = query.model_copy(update={"question": question, "top_k": fetch})
                return await self.retriever.search(sub, searcher)

        results = await asyncio.gather(*(search(q) for q in plan.subqueries))
        fused = reciprocal_rank_fusion(list(results), k=self.settings.planner_rrf_k, top_k=fetch)
        return self.planner.prefer(plan, fused)[: query.top_k]

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
        refused = False
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
                        external=(
                            ExternalSession(
                                self.external,
                                principal.model_copy(deep=True),
                                self.external.as_of(query.as_of),
                                self.external.providers(principal),
                            )
                            if self.external is not None and principal is not None
                            else None
                        ),
                    ),
                )
                content, citations, tool_count = (
                    state["content"],
                    state["citations"],
                    state["tool_count"],
                )
                calculations, tool_errors = state["calculations"], state["tool_errors"]
                verification = state["verification"]
                refused = state.get("refused", False)
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
                _, plan = recall(query)
                refused = bool(
                    plan is not None
                    and plan.decision.route == "refuse_out_of_scope"
                    and plan.skip_retrieval
                    and not hits
                )
            decision = self.abstention.decide(
                content=content,
                hits=hits,
                verification=verification,
                citation_coverage=citation_coverage(content) if content else None,
                tool_calls=tool_count,
                tool_errors=tool_errors,
                out_of_scope=refused,
            )
            if decision.abstained:
                # Nothing that points at evidence or echoes the rejected draft is emitted;
                # the decision keeps client-safe reason codes and the unsupported count.
                content, citations, calculations = "", [], []
                if verification is not None:
                    verification = verification.model_copy(update={"unsupported_numbers": []})
            span.set_attribute("rag.tool_calls", tool_count)
            span.set_attribute("rag.abstained", decision.abstained)
            if verification is not None:
                span.set_attribute("rag.verification", verification.status)
        elapsed = (time.perf_counter() - start) * 1000
        generation_ms = elapsed if hits or (query.profile == Profile.CHAT and principal) else 0
        calculations = calculations[-MAX_CALCULATIONS:]
        # One validated construction: bounds and evidence references are checked before
        # anything is emitted (model_copy elsewhere skips validation, so dump first).
        return Answer.model_validate(
            {
                "answer": content or ABSTENTION,
                "citations": [citation.model_dump() for citation in citations],
                "abstained": not bool(content),
                "profile": query.profile,
                "retrieval_ms": round(retrieval_ms, 3),
                "generation_ms": round(generation_ms, 3),
                "request_id": request_id,
                "model": self.provider.model,
                "tool_calls": tool_count,
                "calculation": calculations[-1].model_dump() if calculations else None,
                "calculations": [calculation.model_dump() for calculation in calculations],
                "verification": verification.model_dump() if verification else None,
                "abstention": decision.model_dump(),
            }
        )
