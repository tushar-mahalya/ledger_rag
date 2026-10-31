"""Energy-based endpointing over 20 ms frames.

Deliberately simple and dependency-free. It is adequate for telephony narrowband
audio with a sensible RMS threshold; noisy environments benefit from a neural VAD
(e.g. Silero) behind the same ``process`` contract.
"""

from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from ledger_rag.config import Settings
from ledger_rag.voice.codec import rms

FRAME_MS = 20


@dataclass(frozen=True)
class VadConfig:
    speech_rms: int = 500
    min_speech_ms: int = 200
    barge_in_ms: int = 300
    trailing_silence_ms: int = 500
    max_utterance_seconds: float = 15.0
    preroll_ms: int = 200

    @classmethod
    def from_settings(cls, settings: Settings) -> "VadConfig":
        return cls(
            speech_rms=settings.vad_speech_rms,
            min_speech_ms=settings.vad_min_speech_ms,
            barge_in_ms=settings.vad_barge_in_ms,
            trailing_silence_ms=settings.vad_trailing_silence_ms,
            max_utterance_seconds=settings.vad_max_utterance_seconds,
        )


class VadEventKind(StrEnum):
    SPEECH_START = "speech_start"
    BARGE_IN = "barge_in"
    SPEECH_END = "speech_end"


@dataclass(frozen=True)
class VadEvent:
    kind: VadEventKind
    audio: bytes = b""
    capped: bool = False


class EnergyVad:
    def __init__(self, config: VadConfig, sample_rate: int) -> None:
        self.config = config
        self.sample_rate = sample_rate
        self.frame_bytes = sample_rate * FRAME_MS // 1000 * 2
        self.max_utterance_bytes = int(config.max_utterance_seconds * sample_rate) * 2
        self._pending = bytearray()
        self._preroll: deque[bytes] = deque(maxlen=max(1, config.preroll_ms // FRAME_MS))
        self._onset: list[bytes] = []
        self._utterance = bytearray()
        self._speech_ms = 0
        self._silence_ms = 0
        self.in_utterance = False

    def reset(self) -> None:
        self._pending.clear()
        self._end_utterance()

    def _end_utterance(self) -> None:
        self._preroll.clear()
        self._onset = []
        self._utterance = bytearray()
        self._speech_ms = 0
        self._silence_ms = 0
        self.in_utterance = False

    def process(self, pcm: bytes, playing: bool = False) -> list[VadEvent]:
        """Consume PCM16 audio. ``playing`` marks that agent audio is audible (barge-in)."""
        self._pending.extend(pcm)
        events: list[VadEvent] = []
        size = self.frame_bytes
        while len(self._pending) >= size:
            frame = bytes(self._pending[:size])
            del self._pending[:size]
            event = self._frame(frame, playing)
            if event is not None:
                events.append(event)
        return events

    def flush(self) -> VadEvent | None:
        """Force end-of-utterance (push-to-talk / client ``end`` control)."""
        if not self.in_utterance:
            self.reset()
            return None
        self._utterance.extend(self._pending)
        audio = bytes(self._utterance[: self.max_utterance_bytes])
        self.reset()
        return VadEvent(VadEventKind.SPEECH_END, audio)

    def _frame(self, frame: bytes, playing: bool) -> VadEvent | None:
        speech = rms(frame) >= self.config.speech_rms
        if not self.in_utterance:
            if not speech:
                self._preroll.extend(self._onset)
                self._onset = []
                self._speech_ms = 0
                self._preroll.append(frame)
                return None
            self._onset.append(frame)
            self._speech_ms += FRAME_MS
            needed = self.config.barge_in_ms if playing else self.config.min_speech_ms
            if self._speech_ms < needed:
                return None
            self.in_utterance = True
            self._silence_ms = 0
            self._utterance = bytearray(b"".join([*self._preroll, *self._onset]))
            self._preroll.clear()
            self._onset = []
            return VadEvent(VadEventKind.BARGE_IN if playing else VadEventKind.SPEECH_START)
        self._utterance.extend(frame)
        self._silence_ms = 0 if speech else self._silence_ms + FRAME_MS
        if self._silence_ms >= self.config.trailing_silence_ms:
            # Keep ~100 ms of trailing silence; STT models handle a clean tail better.
            trim = max(0, self._silence_ms - 100) * self.sample_rate // 1000 * 2
            audio = bytes(self._utterance[: len(self._utterance) - trim])
            self._end_utterance()
            return VadEvent(VadEventKind.SPEECH_END, audio)
        if len(self._utterance) >= self.max_utterance_bytes:
            audio = bytes(self._utterance[: self.max_utterance_bytes])
            self._end_utterance()
            return VadEvent(VadEventKind.SPEECH_END, audio, capped=True)
        return None
