from typing import Literal

ErrorKind = Literal[
    "disabled",
    "forbidden",
    "invalid_request",
    "not_found",
    "timeout",
    "rate_limited",
    "too_large",
    "upstream",
    "circuit_open",
    "budget_exhausted",
]

# Kinds that say something about provider health and therefore trip the circuit breaker.
FAILURE_KINDS: frozenset[ErrorKind] = frozenset(
    {"timeout", "rate_limited", "upstream", "too_large"}
)


class ExternalDataError(Exception):
    """A sanitized failure; ``message`` never contains URLs, keys or upstream bodies."""

    def __init__(self, kind: ErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind: ErrorKind = kind
        self.message = message
