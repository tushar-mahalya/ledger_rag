"""Self-consistency: independent samples vote on the executed answer, not the text.

Programs that differ syntactically but compute the same value agree. Invalid,
unparseable or ungrounded samples carry no vote. The winning share of valid votes
is a calibration signal for abstention and risk-coverage reporting.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

Value = float | str


def vote_key(value: Value) -> str:
    """Bucket executed answers: equal after rounding to 5 decimals (official precision)."""
    if isinstance(value, str):
        return value
    rounded = round(value, 5)
    return repr(0.0 if rounded == 0 else rounded)


@dataclass(frozen=True)
class Candidate:
    """One sample's outcome. ``value`` is None when it produced no valid answer."""

    program: str | None
    value: Value | None
    reason: str = "ok"


@dataclass(frozen=True)
class Decision:
    program: str | None
    value: Value | None
    votes: int
    valid: int
    total: int

    @property
    def agreement(self) -> float:
        """Winning votes over all samples (invalid samples count against confidence)."""
        return self.votes / self.total if self.total else 0.0


def decide(candidates: Sequence[Candidate]) -> Decision:
    """Majority over executed values; ties go to the earliest sample (greedy first)."""
    valid = [c for c in candidates if c.value is not None]
    if not valid:
        return Decision(None, None, 0, 0, len(candidates))
    counts = Counter(vote_key(c.value) for c in valid if c.value is not None)
    best = max(counts.values())
    for candidate in valid:
        if candidate.value is not None and counts[vote_key(candidate.value)] == best:
            return Decision(candidate.program, candidate.value, best, len(valid), len(candidates))
    raise AssertionError("unreachable: a valid candidate holds the maximum")
