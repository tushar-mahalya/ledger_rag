"""Explicit abstention policy over observable answer features.

The default thresholds reproduce the previous rule (abstain only without validated
content) plus enforced verification failures. They are NOT calibrated: fit them on
FinQA dev data (never the held-out test split) before treating them as risk controls.
Every decision carries its feature values so the evaluation workstream can fit them.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from ledger_rag.core.config import Settings
from ledger_rag.core.models import AbstentionDecision, AbstentionFeatures, Hit, Verification


@dataclass(frozen=True)
class AbstentionPolicy:
    min_top_score: float | None = None
    min_citation_coverage: float = 0.0
    max_tool_errors: int | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "AbstentionPolicy":
        return cls(
            min_top_score=settings.abstain_min_top_score,
            min_citation_coverage=settings.abstain_min_citation_coverage,
            max_tool_errors=settings.abstain_max_tool_errors,
        )

    def decide(
        self,
        *,
        content: str,
        hits: Sequence[Hit],
        verification: Verification | None,
        citation_coverage: float | None,
        tool_calls: int,
        tool_errors: int,
    ) -> AbstentionDecision:
        scores = [hit.score for hit in hits]
        features = AbstentionFeatures(
            hit_count=len(hits),
            top_score=round(max(scores), 6) if scores else None,
            mean_score=round(sum(scores) / len(scores), 6) if scores else None,
            citation_coverage=citation_coverage,
            checked_numbers=verification.checked_numbers if verification else 0,
            unsupported_numbers=len(verification.unsupported_numbers) if verification else 0,
            verification_status=verification.status if verification else "skipped",
            verifier_score=verification.verifier_score if verification else None,
            tool_calls=tool_calls,
            tool_errors=tool_errors,
        )
        reasons: list[str] = []
        if not content:
            reasons.append("no_validated_answer")
        if features.verification_status == "failed":
            reasons.append("verification_failed")
        if self.min_top_score is not None and (
            features.top_score is None or features.top_score < self.min_top_score
        ):
            reasons.append("retrieval_score_below_threshold")
        if content and (citation_coverage or 0.0) < self.min_citation_coverage:
            reasons.append("citation_coverage_below_threshold")
        if self.max_tool_errors is not None and tool_errors > self.max_tool_errors:
            reasons.append("tool_errors_above_threshold")
        return AbstentionDecision(abstained=bool(reasons), reasons=reasons, features=features)
