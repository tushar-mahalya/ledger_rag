"""Client for the Ledger engine's public contract (``POST /v1/query``, ``profile=voice``).

Voice never imports engine internals: it validates the documented response shape
itself (checked against generated engine fixtures in tests) and speaks only the
validated ``answer`` field or a fixed phrase.
"""

import re
import time
from dataclasses import dataclass
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ledger_voice.config import VoiceSettings

# Fixed phrases: the only text voice speaks that did not come from a validated answer.
ABSTENTION_PROMPT = "I'm sorry, I couldn't find that in the documents I'm allowed to use."
TOO_LONG_PROMPT = (
    "That answer is too detailed to read out over the phone. Please use the chat workspace."
)
BUSY_PROMPT = "I'm sorry, I'm handling a lot of requests right now. Please try again in a moment."
FAILURE_PROMPT = "I'm sorry, something went wrong on my side. Please try again in a moment."

_CITATION = re.compile(r"\s*\[C\d+\]")
_MARKUP = re.compile(r"[*_#`>|~]+")
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([.,;:!?%])")
# Frequent STT output on noise; never worth an engine round trip.
_NOISE = {"you", "thank you", "thanks for watching", "thank you for watching", "uh", "um", "hmm"}

Outcome = Literal["answered", "abstained", "too_long", "busy", "failed"]


class EngineAnswer(BaseModel):
    """The subset of the engine ``Answer`` contract voice relies on."""

    model_config = ConfigDict(extra="ignore")

    answer: str = Field(max_length=900_000)
    abstained: bool
    profile: Literal["voice", "chat"]
    request_id: str = Field(max_length=200)


@dataclass(frozen=True)
class Reply:
    text: str
    outcome: Outcome
    engine_ms: float
    request_id: str | None = None


def is_meaningful(transcript: str) -> bool:
    normalized = " ".join(re.sub(r"[^\w\s]", " ", transcript.lower()).split())
    return sum(char.isalnum() for char in normalized) >= 2 and normalized not in _NOISE


def to_speakable(answer: EngineAnswer, max_chars: int) -> tuple[str, Outcome]:
    """Strip citation markers and markup; never cut a validated sentence in half.

    An answer longer than ``max_chars`` is shortened only at a sentence boundary that
    keeps at least one complete sentence; otherwise the caller hears TOO_LONG_PROMPT.
    """
    if answer.abstained or answer.profile != "voice":
        return ABSTENTION_PROMPT, "abstained"
    text = " ".join(_MARKUP.sub(" ", _CITATION.sub("", answer.answer)).split())
    text = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)
    if not text:
        return ABSTENTION_PROMPT, "abstained"
    if len(text) <= max_chars:
        return text, "answered"
    ends = [m.end() for m in _SENTENCE_END.finditer(text) if m.end() <= max_chars]
    if not ends:
        return TOO_LONG_PROMPT, "too_long"
    return text[: ends[-1]].strip(), "answered"


class EngineClient:
    """Keeps one HTTP/1.1 keep-alive pool to the engine for the service lifetime."""

    def __init__(
        self, settings: VoiceSettings, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._settings = settings
        token = settings.engine_token.get_secret_value() if settings.engine_token else ""
        self._client = httpx.AsyncClient(
            base_url=settings.engine_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=httpx.Timeout(settings.engine_timeout_seconds, connect=2.0),
            follow_redirects=False,
            transport=transport,
            limits=httpx.Limits(max_keepalive_connections=8, keepalive_expiry=60),
        )

    async def answer(self, transcript: str) -> Reply:
        question = transcript.strip()[: self._settings.max_transcript_chars]
        start = time.perf_counter()
        try:
            response = await self._client.post(
                "/v1/query", json={"question": question, "profile": "voice"}
            )
        except httpx.HTTPError:
            return Reply(FAILURE_PROMPT, "failed", _ms(start))
        elapsed = _ms(start)
        if response.status_code == 429:
            return Reply(BUSY_PROMPT, "busy", elapsed)
        if response.status_code != 200:
            return Reply(FAILURE_PROMPT, "failed", elapsed)
        try:
            answer = EngineAnswer.model_validate_json(response.content)
        except ValidationError:
            return Reply(FAILURE_PROMPT, "failed", elapsed)
        text, outcome = to_speakable(answer, self._settings.max_spoken_chars)
        return Reply(text, outcome, elapsed, answer.request_id)

    async def close(self) -> None:
        await self._client.aclose()


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
