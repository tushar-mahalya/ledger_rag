"""Query planner: analysis, decision, rewrite and decomposition for one request."""

import asyncio
import hashlib
import time
from contextvars import ContextVar
from dataclasses import dataclass, replace

import structlog

from ledger_rag.core.config import Settings
from ledger_rag.core.contracts import Provider
from ledger_rag.core.models import Hit, Principal, Profile, QueryRequest
from ledger_rag.planning.analysis import Gazetteer, QueryAnalysis, analyze
from ledger_rag.planning.decisions import (
    Decision,
    DecisionEngine,
    HeuristicDecisionEngine,
    LayaDecisionEngine,
)
from ledger_rag.planning.decomposition import llm_subqueries, template_subqueries
from ledger_rag.planning.rewrite import (
    RewriteMethod,
    carry_over,
    deterministic_rewrite,
    rewrite,
)

logger = structlog.get_logger()


@dataclass(frozen=True)
class QueryPlan:
    fingerprint: str
    analysis: QueryAnalysis
    decision: Decision
    retrieval_question: str
    rewrite_method: RewriteMethod
    subqueries: tuple[str, ...]
    elapsed_ms: float = 0.0

    @property
    def skip_retrieval(self) -> bool:
        return not self.subqueries

    def attributes(self) -> dict[str, str | int | float | bool]:
        """Span attributes. Never includes raw user text, history or rewritten queries."""
        return {
            "rag.plan.route": self.decision.route,
            "rag.plan.decision_source": self.decision.source,
            "rag.plan.decision_confidence": self.decision.confidence,
            "rag.plan.operation": self.analysis.operation,
            "rag.plan.needs_calculation": self.decision.needs_calculation,
            "rag.plan.decompose": self.decision.decompose,
            "rag.plan.needs_handoff": self.decision.needs_handoff,
            "rag.plan.rewrite": self.rewrite_method,
            "rag.plan.subqueries": len(self.subqueries),
            "rag.plan.companies": len(self.analysis.companies),
            "rag.plan.years": len(self.analysis.years),
            "rag.plan.metrics": len(self.analysis.metrics),
            "rag.plan.analysis_confidence": self.analysis.confidence,
            "rag.plan.ms": round(self.elapsed_ms, 3),
        }


def fingerprint(query: QueryRequest) -> str:
    digest = hashlib.sha256()
    digest.update(query.profile.value.encode())
    digest.update(b"\0" + query.question.encode())
    for message in query.history:
        digest.update(b"\0" + message.role.encode() + b"\0" + message.content.encode())
    return digest.hexdigest()


# The plan made during retrieval, read by the chat graph's plan node in the same request
# context. ``(fingerprint, None)`` records that planning was disabled for that request.
_CURRENT: ContextVar[tuple[str, QueryPlan | None] | None] = ContextVar(
    "ledger_query_plan", default=None
)


def remember(query: QueryRequest, plan: QueryPlan | None) -> None:
    _CURRENT.set((fingerprint(query), plan))


def recall(query: QueryRequest) -> tuple[bool, QueryPlan | None]:
    """Return ``(found, plan)`` for this exact request; ``plan`` is None when disabled."""
    current = _CURRENT.get()
    if current is None or current[0] != fingerprint(query):
        return False, None
    return True, current[1]


class Planner:
    def __init__(
        self,
        settings: Settings,
        *,
        engine: DecisionEngine | None = None,
        provider: Provider | None = None,
        gazetteer: Gazetteer | None = None,
    ) -> None:
        self.settings = settings
        self.engine = engine or HeuristicDecisionEngine()
        self.heuristic = HeuristicDecisionEngine()
        # None selects deterministic rewriting/decomposition only.
        self.provider = provider
        self.gazetteer = gazetteer or Gazetteer()

    def _gazetteer(self, principal: Principal | None) -> Gazetteer:
        return self.gazetteer.with_principal(principal, self.settings.planner_company_group_prefix)

    def _routing(self, query: QueryRequest, gazetteer: Gazetteer) -> QueryAnalysis:
        """Current-question analysis plus entities carried from history (never raw text)."""
        analysis = analyze(query.question, gazetteer)
        if query.history and analysis.follow_up and not analysis.self_contained:
            return carry_over(query.question, analysis, query.history, gazetteer).analysis
        return analysis

    def _limit(self, profile: Profile) -> int:
        if profile == Profile.VOICE:
            return self.settings.planner_voice_max_subqueries
        return self.settings.planner_max_subqueries

    def _finish(
        self,
        query: QueryRequest,
        analysis: QueryAnalysis,
        decision: Decision,
        question: str,
        method: RewriteMethod,
        subqueries: list[str],
        start: float,
    ) -> QueryPlan:
        decision = replace(
            decision,
            needs_calculation=decision.needs_calculation or analysis.needs_calculation,
            decompose=decision.decompose or analysis.multi_part,
        )
        refuse = (
            decision.route == "refuse_out_of_scope"
            and decision.confidence >= self.settings.decision_min_confidence
        )
        return QueryPlan(
            fingerprint=fingerprint(query),
            analysis=analysis,
            decision=decision,
            retrieval_question=question,
            rewrite_method=method,
            subqueries=() if refuse else tuple(dict.fromkeys(subqueries)),
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )

    def deterministic(self, query: QueryRequest, principal: Principal | None = None) -> QueryPlan:
        """Offline plan with no network or model calls (voice fast path, graph fallback)."""
        start = time.perf_counter()
        gazetteer = self._gazetteer(principal)
        analysis = analyze(query.question, gazetteer)
        question = query.question
        method: RewriteMethod = "none"
        if query.history and analysis.follow_up and not analysis.self_contained:
            question = deterministic_rewrite(
                query.question,
                analysis,
                query.history,
                gazetteer,
                self.settings.planner_rewrite_max_chars,
            )
            method = "deterministic"
            analysis = analyze(question, gazetteer)
        routing = self._routing(query, gazetteer)
        decision = self.heuristic.decide_sync(query.question, routing, query.profile)
        subqueries = [question]
        if self.settings.planner_decompose and (decision.decompose or analysis.multi_part):
            subqueries = template_subqueries(question, analysis, self._limit(query.profile))
        return self._finish(query, analysis, decision, question, method, subqueries, start)

    async def plan(self, query: QueryRequest, principal: Principal | None = None) -> QueryPlan:
        start = time.perf_counter()
        gazetteer = self._gazetteer(principal)
        voice = query.profile == Profile.VOICE
        base = self.deterministic(query, principal)
        provider = None if voice else self.provider
        initial = analyze(query.question, gazetteer)
        # Routing sees only the current question plus structured carried entities; the
        # rewritten (history-bearing) question is used for retrieval only.
        decision, rewritten = await asyncio.gather(
            self.engine.decide(query.question, self._routing(query, gazetteer), query.profile),
            rewrite(
                query.question,
                initial,
                query.history,
                gazetteer,
                provider=provider,
                deadline_seconds=self.settings.planner_rewrite_timeout_seconds,
                max_chars=self.settings.planner_rewrite_max_chars,
            ),
        )
        analysis = (
            base.analysis
            if rewritten.question == base.retrieval_question
            else analyze(rewritten.question, gazetteer)
        )
        subqueries = [rewritten.question]
        limit = self._limit(query.profile)
        if self.settings.planner_decompose and (decision.decompose or analysis.multi_part):
            subqueries = template_subqueries(rewritten.question, analysis, limit)
            if (
                len(subqueries) == 1
                and provider is not None
                and self.settings.planner_llm_decompose
            ):
                try:
                    subqueries += await llm_subqueries(
                        provider,
                        rewritten.question,
                        deadline_seconds=self.settings.planner_rewrite_timeout_seconds,
                        limit=limit,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # Optional enrichment; the template plan stands.
                    logger.warning("decompose_fallback", reason=type(exc).__name__)
        return self._finish(
            query, analysis, decision, rewritten.question, rewritten.method, subqueries, start
        )

    def narrow(self, plan: QueryPlan, principal: Principal) -> Principal:
        """Optionally restrict company groups to the named companies; never adds access."""
        prefix = self.settings.planner_company_group_prefix
        if not self.settings.planner_hard_company_filter or not plan.analysis.companies:
            return principal
        wanted = {f"{prefix}{company}" for company in plan.analysis.companies}
        if not wanted <= set(principal.groups):
            return principal
        groups = [g for g in principal.groups if not g.startswith(prefix) or g in wanted]
        return principal.model_copy(update={"groups": groups})

    def prefer(self, plan: QueryPlan, hits: list[Hit]) -> list[Hit]:
        """Stable soft constraint: named companies first, then matching report years.

        Only gazetteer-recognised companies (high precision) trigger reordering; years alone
        never do, because 10-Ks restate prior periods.
        """
        companies, years = plan.analysis.companies, plan.analysis.years
        if not companies:
            return hits
        prefix = self.settings.planner_company_group_prefix

        def score(hit: Hit) -> int:
            chunk = hit.chunk
            company = any(
                chunk.document_id.startswith(f"{c}/")
                or chunk.title.startswith(f"{c} ")
                or f"{prefix}{c}" in chunk.allowed_groups
                for c in companies
            )
            # A 10-K for report year Y discloses Y and up to two prior years.
            year = any(0 <= chunk.published_at.year - y <= 2 for y in years)
            return 2 * company + year

        return sorted(hits, key=lambda hit: -score(hit))

    async def close(self) -> None:
        await self.engine.close()


def make_planner(settings: Settings, provider: Provider | None) -> Planner:
    engine: DecisionEngine
    if settings.decision_backend == "laya" and settings.decision_api_key is not None:
        engine = LayaDecisionEngine(
            settings.decision_url,
            settings.decision_api_key,
            chat_timeout=settings.decision_chat_timeout_seconds,
            voice_timeout=settings.decision_voice_timeout_seconds,
            min_confidence=settings.decision_min_confidence,
        )
    else:
        engine = HeuristicDecisionEngine()
    gazetteer = (
        Gazetteer.from_file(settings.planner_gazetteer_path)
        if settings.planner_gazetteer_path
        else Gazetteer()
    )
    llm = (
        provider
        if settings.planner_rewrite_mode == "llm" and settings.provider != "extractive"
        else None
    )
    return Planner(settings, engine=engine, provider=llm, gazetteer=gazetteer)
