"""Routing and turn decisions: deterministic heuristics and an optional Laya HTTP backend."""

import asyncio
import re
import time
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

import httpx
import structlog
from pydantic import SecretStr

from ledger_rag.core.models import Profile
from ledger_rag.planning.analysis import QueryAnalysis
from ledger_rag.planning.screening import ADVICE, INJECTION, OFF_TOPIC

logger = structlog.get_logger()

Route = Literal["retrieve", "direct_answer", "clarify", "refuse_out_of_scope"]
ROUTES: tuple[Route, ...] = ("retrieve", "direct_answer", "clarify", "refuse_out_of_scope")
DecisionSource = Literal["heuristic", "laya", "heuristic_fallback"]

_GREETING = re.compile(
    r"^\s*(?:hi|hello|hey|thanks|thank you|thx|good (?:morning|afternoon|evening)|bye|goodbye"
    r"|ok(?:ay)?|got it|cheers|please hold(?: on)?(?: a (?:second|moment|minute))?"
    r"|what can you do|who are you|how do(?:es)? (?:this|you) work)\b[\s!.?,]*"
    r"(?:that is all|that's all|there|a (?:second|moment|minute))?[\s!.?]*$",
    re.I,
)
_HANDOFF = re.compile(
    r"\b(?:speak|talk|transfer|connect|put) (?:me )?(?:to|with|through to) (?:a |an |the )?"
    r"(?:human|person|agent|representative|operator|someone)\b|\b(?:representative|operator)\b"
    r"|\b(?:this is (?:useless|ridiculous)|you'?re not helping)\b",
    re.I,
)


@dataclass(frozen=True)
class Decision:
    route: Route
    needs_calculation: bool
    decompose: bool
    needs_handoff: bool = False
    confidence: float = 0.5
    source: DecisionSource = "heuristic"


class DecisionEngine(Protocol):
    async def decide(self, text: str, analysis: QueryAnalysis, profile: Profile) -> Decision: ...

    async def close(self) -> None: ...


class HeuristicDecisionEngine:
    """Offline, deterministic default. Refusals require explicit, high-precision patterns.

    ``text`` must be the current question only, never history or a rewritten query;
    ``analysis`` may include structured entities carried over from history.
    """

    def decide_sync(self, text: str, analysis: QueryAnalysis, profile: Profile) -> Decision:
        has_finance = bool(analysis.metrics or analysis.companies or analysis.years)
        # Entities carried from history resolve follow-ups but cannot excuse an off-topic or
        # small-talk *current* question.
        own_finance = has_finance and not analysis.carried_over
        route: Route
        if INJECTION.search(text) or ADVICE.search(text):
            route, confidence = "refuse_out_of_scope", 0.9
        elif OFF_TOPIC.search(text) and not own_finance:
            route, confidence = "refuse_out_of_scope", 0.85
        elif _GREETING.match(text) and not own_finance:
            route, confidence = "direct_answer", 0.85
        elif not has_finance and analysis.follow_up and len(text.split()) <= 4:
            route, confidence = "clarify", 0.6
        else:
            route, confidence = "retrieve", 0.7 if has_finance else 0.5
        return Decision(
            route=route,
            needs_calculation=analysis.needs_calculation,
            decompose=analysis.multi_part,
            needs_handoff=profile == Profile.VOICE and bool(_HANDOFF.search(text)),
            confidence=confidence,
        )

    async def decide(self, text: str, analysis: QueryAnalysis, profile: Profile) -> Decision:
        return self.decide_sync(text, analysis, profile)

    async def close(self) -> None:
        return None


class LayaDecisionEngine:
    """Client for compute/decisions.py with a strict deadline and heuristic fallback."""

    def __init__(
        self,
        url: str,
        api_key: SecretStr,
        *,
        chat_timeout: float,
        voice_timeout: float,
        min_confidence: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.url = url.rstrip("/") + "/decide"
        self.chat_timeout, self.voice_timeout = chat_timeout, voice_timeout
        self.min_confidence = min_confidence
        self.heuristic = HeuristicDecisionEngine()
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key.get_secret_value()}"},
            timeout=httpx.Timeout(max(chat_timeout, voice_timeout)),
            transport=transport,
        )

    async def _remote(self, text: str, profile: Profile, deadline: float) -> dict[str, Any]:
        async with asyncio.timeout(deadline):
            response = await self._client.post(
                self.url,
                json={"text": text[:2000], "profile": profile.value},
                timeout=httpx.Timeout(deadline),
            )
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict) or payload.get("route") not in ROUTES:
            raise ValueError("Invalid decision payload")
        return payload

    async def decide(self, text: str, analysis: QueryAnalysis, profile: Profile) -> Decision:
        fallback = self.heuristic.decide_sync(text, analysis, profile)
        deadline = self.voice_timeout if profile == Profile.VOICE else self.chat_timeout
        start = time.perf_counter()
        try:
            payload = await self._remote(text, profile, deadline)
            route_confidence = float(payload["route_confidence"])
            calculation = float(payload["needs_calculation"])
            decompose = float(payload["decompose"])
            handoff = float(payload["needs_handoff"])
        except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "decision_fallback",
                backend="laya",
                reason=type(exc).__name__,
                elapsed_ms=round((time.perf_counter() - start) * 1000, 1),
            )
            return replace(fallback, source="heuristic_fallback")
        # Explicit high-precision heuristic refusals always win over a model route.
        route: Route
        if fallback.route == "refuse_out_of_scope" and fallback.confidence >= 0.85:
            route, confidence = fallback.route, fallback.confidence
        elif route_confidence >= self.min_confidence and not (
            # Smoke runs showed zero-shot Laya refusing elliptical follow-ups ("What about
            # 2010?"); a query with financial entities is never refused/answered by the model.
            payload["route"] != "retrieve"
            and (analysis.metrics or analysis.years or analysis.companies)
        ):
            route, confidence = payload["route"], route_confidence
        else:
            route, confidence = fallback.route, fallback.confidence
        return Decision(
            route=route,
            # Deterministic operation detection is reliable; the model may only add to it.
            needs_calculation=fallback.needs_calculation or calculation >= 0.5,
            # Decomposition needs extracted entities to build sub-queries from.
            decompose=fallback.decompose
            or (decompose >= 0.5 and len(analysis.companies) + len(analysis.years) > 1),
            needs_handoff=profile == Profile.VOICE and (fallback.needs_handoff or handoff >= 0.5),
            confidence=round(confidence, 4),
            source="laya",
        )

    async def close(self) -> None:
        await self._client.aclose()
