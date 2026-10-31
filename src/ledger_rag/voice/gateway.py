"""Transport-agnostic voice turn state machine.

LISTENING -> TRANSCRIBING -> THINKING -> SPEAKING -> LISTENING

Inbound PCM is fed through the VAD on the receive loop; each completed utterance runs as
one cancellable asyncio task (STT -> RAG -> TTS). Caller speech while the agent is
speaking cancels the task and emits a ``ClearEvent`` (barge-in).
"""

import asyncio
import re
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import httpx
import structlog
from fastapi import HTTPException
from opentelemetry import trace

from ledger_rag.core.config import Settings
from ledger_rag.core.models import Answer, Principal, Profile, QueryRequest
from ledger_rag.core.retry import request_deadline
from ledger_rag.generation.providers import ProviderError
from ledger_rag.observability.telemetry import Metrics
from ledger_rag.orchestration.service import RagService
from ledger_rag.security.auth import RateLimiter
from ledger_rag.voice.codec import Resampler
from ledger_rag.voice.metrics import VoiceMetrics
from ledger_rag.voice.speech import SpeechError, SpeechToText, TextToSpeech
from ledger_rag.voice.vad import EnergyVad, VadConfig, VadEvent, VadEventKind

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)

ABSTENTION_PROMPT = (
    "I'm sorry, I couldn't find that in the documents I'm allowed to use. "
    "I can transfer you to an agent if you'd like."
)
BUSY_PROMPT = (
    "I'm sorry, I'm handling a lot of requests right now. "
    "Please try again in a moment, or I can transfer you to an agent."
)
FAILURE_PROMPT = (
    "I'm sorry, something went wrong on my side. I can transfer you to an agent if you'd like."
)
_CITATION = re.compile(r"\s*\[C\d+\]")
_MARKUP = re.compile(r"[*_#`>|~]+")
# Common Whisper hallucinations on silence/noise; never worth a RAG round trip.
_NOISE = {"you", "thank you", "thanks for watching", "thank you for watching", "uh", "um", "hmm"}
_RECOVERABLE = (
    SpeechError,
    ProviderError,
    httpx.HTTPError,
    TimeoutError,
    ValueError,
    RuntimeError,
)
_SLOT_WAIT_SECONDS = 1.0


class VoiceState(StrEnum):
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass
class TurnMetrics:
    turn_id: str
    utterance_ms: float
    capped: bool = False
    stt_ms: float | None = None
    retrieval_ms: float | None = None
    generation_ms: float | None = None
    tts_first_byte_ms: float | None = None
    first_audio_ms: float | None = None
    total_ms: float | None = None
    agent_audio_ms: float = 0.0
    transcript_chars: int = 0
    answer_chars: int = 0
    abstained: bool | None = None
    outcome: str = "pending"

    def as_dict(self) -> dict[str, Any]:
        return {
            key: round(value, 3) if isinstance(value, float) else value
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True)
class TranscriptEvent:
    turn_id: str
    text: str


@dataclass(frozen=True)
class AnswerEvent:
    turn_id: str
    speakable: str
    answer: Answer | None


@dataclass(frozen=True)
class AudioEvent:
    turn_id: str
    pcm: bytes
    sample_rate: int


@dataclass(frozen=True)
class MarkEvent:
    name: str


@dataclass(frozen=True)
class ClearEvent:
    turn_id: str | None


@dataclass(frozen=True)
class TurnMetricsEvent:
    metrics: TurnMetrics


@dataclass(frozen=True)
class ErrorEvent:
    detail: str


VoiceEvent = (
    TranscriptEvent
    | AnswerEvent
    | AudioEvent
    | MarkEvent
    | ClearEvent
    | TurnMetricsEvent
    | ErrorEvent
)
EventSink = Callable[[VoiceEvent], Awaitable[None]]


@dataclass
class VoiceDependencies:
    settings: Settings
    service: RagService
    stt: SpeechToText
    tts: TextToSpeech
    metrics: VoiceMetrics
    rag_metrics: Metrics | None = None
    rag_slots: asyncio.Semaphore | None = None
    limiter: RateLimiter | None = None


def to_speakable(answer: Answer, max_chars: int) -> str:
    """Strip citation markers/markup and keep at most ``max_chars`` (sentence-aligned)."""
    if answer.abstained:
        return ABSTENTION_PROMPT
    text = _CITATION.sub("", answer.answer)
    text = " ".join(_MARKUP.sub(" ", text).split())
    if len(text) > max_chars:
        cut = text[:max_chars]
        end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "), cut.rfind("."))
        cut = cut[: end + 1] if end >= max_chars // 3 else cut.rsplit(" ", 1)[0] + "."
        text = cut.strip()
    return text or ABSTENTION_PROMPT


def is_meaningful(transcript: str) -> bool:
    normalized = " ".join(re.sub(r"[^\w\s]", " ", transcript.lower()).split())
    return sum(char.isalnum() for char in normalized) >= 2 and normalized not in _NOISE


class VoiceSession:
    def __init__(
        self,
        deps: VoiceDependencies,
        principal: Principal,
        sink: EventSink,
        *,
        input_rate: int,
        output_rate: int,
        await_playback_ack: bool,
        transport: str,
        session_id: str | None = None,
    ) -> None:
        self.deps = deps
        self.settings = deps.settings
        self.principal = principal
        self.sink = sink
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.await_playback_ack = await_playback_ack
        self.transport = transport
        self.session_id = session_id or uuid.uuid4().hex
        self.vad = EnergyVad(VadConfig.from_settings(deps.settings), input_rate)
        self.state = VoiceState.LISTENING
        self.turns: deque[TurnMetrics] = deque(maxlen=50)
        self._task: asyncio.Task[None] | None = None
        self._turn_id: str | None = None
        self._turn_audio = b""
        self._spoke = False
        self._carryover = b""
        self._mark: str | None = None
        self._closed = False

    async def feed(self, pcm: bytes) -> None:
        if self._closed:
            return
        for event in self.vad.process(pcm, playing=self.state is VoiceState.SPEAKING):
            await self._on_vad(event)

    async def end_utterance(self) -> None:
        event = self.vad.flush()
        if event is not None:
            await self._on_vad(event)

    async def playback_complete(self, name: str) -> None:
        if name == self._mark and self.state is VoiceState.SPEAKING and not self._active():
            self._mark = None
            self.state = VoiceState.LISTENING

    async def close(self) -> None:
        self._closed = True
        await self._cancel_turn(keep_audio=False)

    def _active(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _on_vad(self, event: VadEvent) -> None:
        if event.kind in {VadEventKind.SPEECH_START, VadEventKind.BARGE_IN}:
            if self.state is VoiceState.SPEAKING:
                await self._barge_in()
            elif self.state is not VoiceState.LISTENING:
                # Caller kept talking after a premature endpoint: merge with the next utterance.
                await self._cancel_turn(keep_audio=True)
        elif event.kind is VadEventKind.SPEECH_END and not self._closed:
            await self._start_turn(event.audio, event.capped)

    async def _cancel_turn(self, keep_audio: bool) -> None:
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.wait({task})
            if keep_audio and not self._spoke:
                self._carryover = self._turn_audio
        self._task = None
        self.state = VoiceState.LISTENING

    async def _barge_in(self) -> None:
        turn_id = self._turn_id
        await self._cancel_turn(keep_audio=False)
        self._mark = None
        self.deps.metrics.barge_ins.inc()
        logger.info("voice_barge_in", session_id=self.session_id, turn_id=turn_id)
        await self.sink(ClearEvent(turn_id))

    async def _start_turn(self, audio: bytes, capped: bool) -> None:
        await self._cancel_turn(keep_audio=True)
        limit = self.vad.max_utterance_bytes
        audio = (self._carryover + audio)[-limit:]
        self._carryover = b""
        self._turn_audio = audio
        self._spoke = False
        self._turn_id = uuid.uuid4().hex[:12]
        self.state = VoiceState.TRANSCRIBING
        self._task = asyncio.create_task(
            self._run_turn(self._turn_id, audio, time.perf_counter(), capped)
        )

    async def _run_turn(self, turn_id: str, audio: bytes, ended: float, capped: bool) -> None:
        metrics = TurnMetrics(
            turn_id=turn_id,
            utterance_ms=len(audio) / 2 / self.input_rate * 1000,
            capped=capped,
        )
        try:
            with tracer.start_as_current_span("voice.turn") as span:
                span.set_attribute("voice.session_id", self.session_id)
                span.set_attribute("voice.turn_id", turn_id)
                span.set_attribute("voice.transport", self.transport)
                try:
                    await self._turn(metrics, audio, ended)
                finally:
                    span.set_attribute("voice.outcome", metrics.outcome)
        except asyncio.CancelledError:
            metrics.outcome = "interrupted"
            self._record(metrics)
            raise
        except Exception as exc:  # a turn must never kill the session silently
            metrics.outcome = "failed"
            logger.error(
                "voice_turn_failure", session_id=self.session_id, error_class=type(exc).__name__
            )
            self.state = VoiceState.LISTENING
            self._record(metrics)
            return
        self._record(metrics)
        if metrics.outcome != "ignored":
            await self.sink(TurnMetricsEvent(metrics))

    async def _turn(self, metrics: TurnMetrics, audio: bytes, ended: float) -> None:
        try:
            with tracer.start_as_current_span("voice.stt") as span:
                transcript = await self.deps.stt.transcribe(audio, self.input_rate)
                span.set_attribute("voice.transcript_chars", len(transcript))
            metrics.stt_ms = (time.perf_counter() - ended) * 1000
            self.deps.metrics.stt.observe(metrics.stt_ms / 1000)
            text = " ".join(transcript.split())
            metrics.transcript_chars = len(text)
            if not is_meaningful(text):
                metrics.outcome = "ignored"
                self.state = VoiceState.LISTENING
                return
            await self.sink(TranscriptEvent(metrics.turn_id, text))
            self.state = VoiceState.THINKING
            speakable, answer = await self._answer(text, metrics)
        except _RECOVERABLE as exc:
            metrics.outcome = "failed"
            logger.error(
                "voice_turn_failure",
                session_id=self.session_id,
                stage=self.state.value,
                error_class=type(exc).__name__,
            )
            await self.sink(ErrorEvent("Voice turn failed"))
            speakable, answer = FAILURE_PROMPT, None
        metrics.answer_chars = len(speakable)
        await self.sink(AnswerEvent(metrics.turn_id, speakable, answer))
        self.state = VoiceState.SPEAKING
        try:
            await self._speak(metrics, speakable, ended)
        except _RECOVERABLE as exc:
            metrics.outcome = "failed"
            logger.error(
                "voice_tts_failure", session_id=self.session_id, error_class=type(exc).__name__
            )
            await self.sink(ErrorEvent("Speech synthesis failed"))
            self.state = VoiceState.LISTENING
            return
        if not self.await_playback_ack:
            self.state = VoiceState.LISTENING

    async def _answer(self, text: str, metrics: TurnMetrics) -> tuple[str, Answer | None]:
        request = QueryRequest(question=text[:4000], profile=Profile.VOICE)
        deps = self.deps
        if deps.limiter is not None:
            try:
                deps.limiter.check(self.principal)
            except HTTPException:
                metrics.outcome = "rate_limited"
                return BUSY_PROMPT, None
        slots = deps.rag_slots
        if slots is not None:
            try:
                await asyncio.wait_for(slots.acquire(), timeout=_SLOT_WAIT_SECONDS)
            except TimeoutError:
                metrics.outcome = "busy"
                return BUSY_PROMPT, None
        try:
            async with request_deadline(self.settings.query_deadline_seconds, voice=True):
                hits, retrieval_ms = await deps.service.retrieve(request, self.principal)
                answer = await deps.service.generate(
                    request, hits, retrieval_ms, f"voice-{metrics.turn_id}", self.principal
                )
        finally:
            if slots is not None:
                slots.release()
        metrics.retrieval_ms = answer.retrieval_ms
        metrics.generation_ms = answer.generation_ms
        metrics.abstained = answer.abstained
        metrics.outcome = "abstained" if answer.abstained else "answered"
        if deps.rag_metrics is not None:
            profile = Profile.VOICE.value
            deps.rag_metrics.retrieval.labels(profile).observe(answer.retrieval_ms / 1000)
            deps.rag_metrics.generation.labels(profile).observe(answer.generation_ms / 1000)
            if answer.abstained:
                deps.rag_metrics.abstentions.labels(profile).inc()
        return to_speakable(answer, self.settings.voice_max_answer_chars), answer

    async def _speak(self, metrics: TurnMetrics, text: str, ended: float) -> None:
        budget = int(self.settings.voice_max_response_seconds * self.output_rate) * 2
        sent = 0
        resampler: Resampler | None = None
        started = time.perf_counter()
        with tracer.start_as_current_span("voice.tts") as span:
            span.set_attribute("voice.text_chars", len(text))
            stream = self.deps.tts.synthesize(text)
            try:
                async for chunk in stream:
                    if metrics.tts_first_byte_ms is None:
                        metrics.tts_first_byte_ms = (time.perf_counter() - started) * 1000
                        self.deps.metrics.tts_first_byte.observe(metrics.tts_first_byte_ms / 1000)
                    if resampler is None or resampler.source_rate != chunk.sample_rate:
                        resampler = Resampler(chunk.sample_rate, self.output_rate)
                    pcm = resampler.process(chunk.pcm)[: budget - sent]
                    if not pcm:
                        if sent >= budget:
                            logger.warning("voice_response_truncated", session_id=self.session_id)
                            break
                        continue
                    if metrics.first_audio_ms is None:
                        metrics.first_audio_ms = (time.perf_counter() - ended) * 1000
                        self.deps.metrics.first_audio.observe(metrics.first_audio_ms / 1000)
                    self._spoke = True
                    sent += len(pcm)
                    await self.sink(AudioEvent(metrics.turn_id, pcm, self.output_rate))
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()
            span.set_attribute("voice.audio_ms", sent / 2 / self.output_rate * 1000)
        metrics.agent_audio_ms = sent / 2 / self.output_rate * 1000
        metrics.total_ms = (time.perf_counter() - ended) * 1000
        self.deps.metrics.turn.observe(metrics.total_ms / 1000)
        self._mark = metrics.turn_id
        await self.sink(MarkEvent(metrics.turn_id))

    def _record(self, metrics: TurnMetrics) -> None:
        self.turns.append(metrics)
        self.deps.metrics.turns.labels(metrics.outcome).inc()
        # Transcripts and answers are PII: only lengths and timings are logged.
        logger.info(
            "voice_turn",
            session_id=self.session_id,
            transport=self.transport,
            **metrics.as_dict(),
        )


__all__ = [
    "ABSTENTION_PROMPT",
    "AnswerEvent",
    "AudioEvent",
    "ClearEvent",
    "ErrorEvent",
    "EventSink",
    "MarkEvent",
    "TranscriptEvent",
    "TurnMetrics",
    "TurnMetricsEvent",
    "VoiceDependencies",
    "VoiceEvent",
    "VoiceSession",
    "VoiceState",
]
