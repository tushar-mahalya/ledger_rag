"""The answer step: turn detection plus the only producer of speech (ADR 0006).

``EngineAnswerProcessor`` extends Pipecat's ``UserTurnProcessor``, so the decision
that a caller's turn is over is made in the same processor that collects the
transcript. Pipecat delivers system frames (turn and VAD events) ahead of queued
data frames (transcripts); deciding the turn end elsewhere would race the last
transcript. When a turn ends it asks the engine once and emits a single
``TTSSpeakFrame`` with the validated answer or a fixed phrase. Because it sits
directly before TTS and refuses to forward any text frame, neither transcripts nor
model tokens can ever be spoken.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from ledger_voice.engine import Reply, is_meaningful


class Answerer(Protocol):
    async def answer(self, transcript: str) -> Reply: ...


ReplyHook = Callable[[str, Reply], Awaitable[None]]


class EngineAnswerProcessor(UserTurnProcessor):
    def __init__(
        self,
        engine: Answerer,
        *,
        max_transcript_chars: int = 600,
        on_reply: ReplyHook | None = None,
        user_turn_strategies: UserTurnStrategies | None = None,
    ) -> None:
        super().__init__(user_turn_strategies=user_turn_strategies or UserTurnStrategies())
        self._engine = engine
        self._max_chars = max_transcript_chars
        self._on_reply = on_reply
        self._parts: list[str] = []
        self._task: asyncio.Task[None] | None = None
        self.add_event_handler("on_user_turn_started", self._turn_started)
        self.add_event_handler("on_user_turn_stopped", self._turn_stopped)

    @property
    def pending(self) -> bool:
        return self._task is not None and not self._task.done()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # Collect before the turn controller sees the frame, so a turn that ends on
        # this transcript already has it.
        if isinstance(frame, TranscriptionFrame):
            self._collect(frame.text)
        elif isinstance(frame, (InterruptionFrame, EndFrame, CancelFrame)):
            self._parts.clear()
            await self._cancel_answer()
        await super().process_frame(frame, direction)

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TextFrame):
            return  # transcripts and any other text never travel on towards TTS
        await super().push_frame(frame, direction)

    async def _turn_started(self, _processor: Any, _strategy: Any) -> None:
        self._parts.clear()
        await self._cancel_answer()

    async def _turn_stopped(self, _processor: Any, _strategy: Any) -> None:
        transcript = " ".join(self._parts).strip()
        self._parts.clear()
        if not is_meaningful(transcript):
            return
        await self._cancel_answer()
        self._task = self.create_task(self._answer(transcript), "engine_answer")

    def _collect(self, text: str) -> None:
        used = sum(len(part) + 1 for part in self._parts)
        room = self._max_chars - used
        if room > 0 and text.strip():
            self._parts.append(text.strip()[:room])

    async def _answer(self, transcript: str) -> None:
        reply = await self._engine.answer(transcript)
        if self._on_reply is not None:
            await self._on_reply(transcript, reply)
        await self.push_frame(TTSSpeakFrame(reply.text, append_to_context=False))

    async def _cancel_answer(self) -> None:
        if self._task is not None and not self._task.done():
            await self.cancel_task(self._task)
        self._task = None
