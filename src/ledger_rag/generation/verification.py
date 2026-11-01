"""Deterministic numeric provenance checks and an optional model-based groundedness verifier.

Policy (documented in docs/CHAT_API.md):

* Every number in the answer, except years, citation markers, list ordinals, dates and
  filing identifiers, must match a number in the evidence cited by its own sentence
  (or, for sentences without markers, by its paragraph), or a calculator step whose
  evidence is cited there.
* Matching is magnitude-based (sign is usually expressed in words, e.g. "a loss of"),
  scale-aware (``$1.2 billion`` == ``1,200 million``) and tolerates the rounding implied
  by the answer's displayed precision, capped at ``max_rounding_ratio`` relative error.
* A scale-free number matches the as-written mantissa of a scaled number, so table
  values reported "in millions" can be restated with or without the scale word.
"""

import asyncio
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol

import httpx
import structlog

from ledger_rag.core.config import Settings
from ledger_rag.core.models import (
    Calculation,
    CalculationKind,
    Citation,
    Profile,
    Verification,
    VerificationReason,
    VerificationStatus,
)
from ledger_rag.generation.program import NumberMention, extract_numbers
from ledger_rag.orchestration.activity import ActivityCallback, activity

logger = structlog.get_logger()
MAX_REPORTED_NUMBERS = 20
_MARKER = re.compile(r"\[(C[0-9]+)\]")
_SENTENCE_END = re.compile(r"[.!?](?:[ \t]*\[C[0-9]+\])*(?=\s|$)")
_PERCENT_UNITS = {"percent", "percentage_points"}
_LIST_PREFIX = re.compile(r"[ \t]*(?:\d{1,2}[.)]|[-*\u2022])[ \t]+")


@dataclass(frozen=True)
class NumericPolicy:
    relative_tolerance: Decimal = Decimal(0)
    max_rounding_ratio: Decimal = Decimal("0.05")


@dataclass(frozen=True)
class _Reference:
    mention: NumberMention | None = None
    program_value: Decimal | None = None
    program_kind: CalculationKind | None = None


@dataclass(frozen=True)
class Segment:
    text: str
    citation_ids: frozenset[str]
    direct: bool


@dataclass
class NumericCheck:
    checked: int = 0
    unsupported: list[str] = field(default_factory=list)
    citation_coverage: float | None = None


def segments(content: str) -> list[Segment]:
    """Split into sentences/lines; each inherits paragraph markers if it has none."""
    result: list[Segment] = []
    for paragraph in re.split(r"\n[ \t]*\n", content):
        paragraph_ids = frozenset(_MARKER.findall(paragraph))
        for line in paragraph.splitlines():
            start = 0
            pieces = []
            prefix = _LIST_PREFIX.match(line)
            for match in _SENTENCE_END.finditer(line, prefix.end() if prefix else 0):
                pieces.append(line[start : match.end()])
                start = match.end()
            pieces.append(line[start:])
            for piece in pieces:
                if not re.search(r"\w", _MARKER.sub("", piece)):
                    continue
                own = frozenset(_MARKER.findall(piece))
                result.append(Segment(piece, own or paragraph_ids, bool(own)))
    return result


def citation_coverage(content: str) -> float | None:
    parts = segments(content)
    if not parts:
        return None
    return round(sum(part.direct for part in parts) / len(parts), 4)


def _half_ulp(mention: NumberMention) -> Decimal:
    return Decimal(5).scaleb(-(mention.decimals + 1))


def _close(answer: Decimal, reference: Decimal, half_ulp: Decimal, policy: NumericPolicy) -> bool:
    answer, reference = abs(answer), abs(reference)
    difference = abs(answer - reference)
    if difference == 0:
        return True
    if difference > max(half_ulp, policy.relative_tolerance * reference):
        return False
    return difference <= policy.max_rounding_ratio * reference


def _compatible(answer: NumberMention, reference: NumberMention) -> bool:
    units = {answer.unit, reference.unit} - {None}
    if units & _PERCENT_UNITS and max(answer.scale, reference.scale) != 1:
        return False  # "15%" is never supported by "15 million".
    return len(units) <= 1 or units <= _PERCENT_UNITS


def matches(answer: NumberMention, reference: _Reference, policy: NumericPolicy) -> bool:
    ulp = _half_ulp(answer)
    if reference.mention is not None:
        evidence = reference.mention
        if not _compatible(answer, evidence):
            return False
        if _close(answer.absolute, evidence.absolute, ulp * answer.scale, policy):
            return True
        unscaled = answer.scale == 1 or evidence.scale == 1
        return unscaled and _close(answer.value, evidence.value, ulp, policy)
    value, kind = reference.program_value, reference.program_kind
    if value is None or kind is None or kind == "boolean":
        return False
    if kind == "percent":
        return answer.unit in _PERCENT_UNITS and _close(answer.value, value, ulp, policy)
    if kind == "ratio":
        if answer.unit in _PERCENT_UNITS:
            return _close(answer.value / 100, value, ulp / 100, policy)
        if answer.unit == "bps":
            return _close(answer.value / 10000, value, ulp / 10000, policy)
        # A bare multiple such as "1.25x"; never a currency or scaled amount.
        return (
            answer.unit is None and answer.scale == 1 and _close(answer.value, value, ulp, policy)
        )
    if answer.unit in _PERCENT_UNITS or answer.unit == "bps":
        return False
    return _close(answer.absolute, value, ulp * answer.scale, policy)


def check_numbers(
    content: str,
    citations: Sequence[Citation],
    calculations: Sequence[Calculation],
    policy: NumericPolicy,
) -> NumericCheck:
    by_id = {citation.id: citation for citation in citations}
    evidence: dict[str, list[_Reference]] = {
        citation.id: [
            _Reference(mention=mention) for mention in extract_numbers(citation.quote, loose=True)
        ]
        for citation in citations
    }
    # Each step supports only sentences citing that step's own evidence lineage.
    programs = [
        (
            frozenset(step.citation_ids),
            [_Reference(program_value=Decimal(step.result), program_kind=step.kind)],
        )
        for calculation in calculations
        for step in calculation.steps
        if step.evidence_backed and step.citation_ids and step.kind not in {None, "boolean"}
    ]
    check = NumericCheck(citation_coverage=citation_coverage(content))
    for segment in segments(content):
        references: list[_Reference] = []
        for citation_id in sorted(segment.citation_ids):
            if citation_id in by_id:
                references.extend(evidence[citation_id])
        for citation_ids, results in programs:
            if citation_ids & segment.citation_ids:
                references.extend(results)
        for mention in extract_numbers(segment.text, include_years=False, loose=True):
            check.checked += 1
            if not any(matches(mention, reference, policy) for reference in references):
                if len(check.unsupported) < MAX_REPORTED_NUMBERS:
                    check.unsupported.append(mention.text[:64])
    return check


@dataclass(frozen=True)
class GroundednessResult:
    score: float
    unsupported_spans: list[str]


class GroundednessVerifier(Protocol):
    name: str

    async def verify(
        self, *, question: str, contexts: list[str], answer: str
    ) -> GroundednessResult: ...

    async def close(self) -> None: ...


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _valid_span(span: Any) -> bool:
    return (
        isinstance(span, dict)
        and _number(span.get("start"))
        and _number(span.get("end"))
        and isinstance(span.get("text", ""), str)
    )


class HttpGroundednessVerifier:
    """Client for compute/verifier.py (LettuceDetect behind bearer auth)."""

    def __init__(self, url: str, api_key: str, timeout_seconds: float, name: str) -> None:
        self.name = name
        self._url = url.rstrip("/") + "/v1/verify"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 2.0)),
            follow_redirects=False,
        )

    async def verify(
        self, *, question: str, contexts: list[str], answer: str
    ) -> GroundednessResult:
        response = await self._client.post(
            self._url,
            json={"question": question, "contexts": contexts, "answer": answer},
            headers=self._headers,
        )
        response.raise_for_status()
        data: Any = response.json()
        score = data["supported_score"]
        spans = data.get("unsupported_spans", [])
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0 <= score <= 1
            or not isinstance(spans, list)
            or not all(_valid_span(span) for span in spans)
        ):
            raise ValueError("Invalid verifier response")
        return GroundednessResult(
            score=float(score),
            unsupported_spans=[str(span.get("text", ""))[:200] for span in spans[:10]],
        )

    async def close(self) -> None:
        await self._client.aclose()


def make_groundedness_verifier(settings: Settings) -> GroundednessVerifier | None:
    if settings.groundedness_verifier == "off" or not settings.groundedness_verifier_url:
        return None
    key = settings.groundedness_verifier_api_key
    return HttpGroundednessVerifier(
        settings.groundedness_verifier_url,
        key.get_secret_value() if key else "",
        settings.groundedness_verifier_timeout_seconds,
        settings.groundedness_verifier,
    )


class AnswerVerifier:
    """Deterministic checks for every profile; the optional model verifier is chat-only."""

    def __init__(self, settings: Settings, model: GroundednessVerifier | None = None) -> None:
        self.mode = settings.verification_mode
        self.policy = NumericPolicy(
            relative_tolerance=Decimal(str(settings.verification_relative_tolerance)),
            max_rounding_ratio=Decimal(str(settings.verification_max_rounding_ratio)),
        )
        self.model = model
        self.min_score = settings.groundedness_verifier_min_score
        self.timeout = settings.groundedness_verifier_timeout_seconds

    def _outcome(self, reasons: Sequence[VerificationReason]) -> VerificationStatus:
        if not reasons:
            return "verified"
        return "failed" if self.mode == "enforce" else "unverified"

    def check(
        self,
        content: str,
        citations: Sequence[Citation],
        calculations: Iterable[Calculation],
    ) -> Verification:
        if self.mode == "off" or not content:
            return Verification(status="skipped", mode=self.mode)
        result = check_numbers(content, citations, list(calculations), self.policy)
        reasons: list[VerificationReason] = ["unsupported_numbers"] if result.unsupported else []
        return Verification(
            status=self._outcome(reasons),
            mode=self.mode,
            checked_numbers=result.checked,
            unsupported_numbers=result.unsupported,
            reasons=reasons,
        )

    async def verify_model(
        self,
        verification: Verification,
        *,
        content: str,
        question: str,
        citations: Sequence[Citation],
        profile: Profile,
        on_activity: ActivityCallback | None,
    ) -> Verification:
        if (
            self.model is None
            or profile != Profile.CHAT
            or verification.status != "verified"
            or not content
        ):
            return verification
        async with activity(
            on_activity,
            "validation",
            kind="grounding_verification",
            span_name="rag.verifier",
        ) as span:
            try:
                async with asyncio.timeout(self.timeout):
                    result = await self.model.verify(
                        question=question,
                        contexts=[citation.quote for citation in citations],
                        answer=_MARKER.sub("", content),
                    )
            except Exception as exc:
                # Fail open to the deterministic result; never block answers on the verifier.
                # Only the exception class is logged: messages may echo remote content.
                logger.warning("groundedness_verifier_unavailable", error_class=type(exc).__name__)
                span.set_attribute("rag.verifier.available", False)
                return verification.model_copy(
                    update={"verifier": self.model.name, "reasons": ["verifier_unavailable"]}
                )
            span.set_attribute("rag.verifier.available", True)
            span.set_attribute("rag.verifier.score", result.score)
        reasons: list[VerificationReason] = (
            [] if result.score >= self.min_score else ["verifier_below_threshold"]
        )
        return verification.model_copy(
            update={
                "status": self._outcome(reasons),
                "reasons": reasons,
                "verifier": self.model.name,
                "verifier_score": round(result.score, 4),
            }
        )

    async def close(self) -> None:
        if self.model is not None:
            await self.model.close()
