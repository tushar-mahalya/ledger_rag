import time
from collections.abc import Callable
from typing import Literal

from ledger_rag.external.errors import FAILURE_KINDS, ExternalDataError

BreakerState = Literal["closed", "open", "half_open"]


class CircuitBreaker:
    """Stop calling a failing provider for ``reset_seconds`` after ``threshold`` failures.

    Only provider-health failures count (timeouts, 429s, 5xx, oversize bodies); a
    caller's bad concept name or an unknown series never opens the circuit. After the
    reset window one probe is admitted; its outcome closes or re-opens the circuit.
    """

    def __init__(
        self,
        provider: str,
        threshold: int,
        reset_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.provider = provider
        self.threshold = threshold
        self.reset_seconds = reset_seconds
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._probing = False

    @property
    def state(self) -> BreakerState:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self.reset_seconds:
            return "half_open"
        return "open"

    def before(self) -> None:
        state = self.state
        if state == "open" or (state == "half_open" and self._probing):
            raise ExternalDataError(
                "circuit_open", f"{self.provider} is temporarily unavailable, try again later"
            )
        if state == "half_open":
            self._probing = True

    def success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._probing = False

    def release(self) -> None:
        """A cancelled call proves nothing; let the next caller probe."""
        self._probing = False

    def failure(self, error: BaseException) -> None:
        if isinstance(error, ExternalDataError) and error.kind not in FAILURE_KINDS:
            # The provider answered; a bad request says nothing about its health.
            if self._probing:
                self.success()
            return
        self._probing = False
        self._failures += 1
        if self._opened_at is not None or self._failures >= self.threshold:
            self._opened_at = self._clock()
