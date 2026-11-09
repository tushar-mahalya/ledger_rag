"""OpenAI-compatible speech adapters (OpenAI, Groq STT, speaches, faster-whisper-server,
Kokoro-FastAPI). Non-retrying (speech is latency-bound), no redirects, sanitized errors."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from ledger_rag.core.config import Settings
from ledger_rag.generation.providers import _endpoint
from ledger_rag.voice.codec import MAX_RATE, MIN_RATE, WavStreamParser, build_wav

MAX_TRANSCRIPT_CHARS = 4000


class SpeechError(RuntimeError):
    """A sanitized STT/TTS transport or protocol failure (never contains upstream bodies)."""


@dataclass(frozen=True)
class AudioChunk:
    pcm: bytes
    sample_rate: int


class SpeechToText(Protocol):
    async def transcribe(self, pcm: bytes, sample_rate: int) -> str: ...

    async def close(self) -> None: ...


class TextToSpeech(Protocol):
    def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        """Yield mono s16le PCM chunks as soon as the upstream produces them."""
        ...

    async def close(self) -> None: ...


def _headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"} if key else {}


class OpenAITranscriber:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.model = settings.stt_model
        self.language = settings.stt_language
        self._url = _endpoint(settings.stt_base_url, "audio/transcriptions")
        self._key = settings.stt_api_key.get_secret_value() if settings.stt_api_key else ""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.stt_timeout_seconds, connect=5.0),
            follow_redirects=False,
            transport=transport,
        )

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        data = {"model": self.model, "response_format": "json"}
        if self.language:
            data["language"] = self.language
        files = {"file": ("utterance.wav", build_wav(pcm, sample_rate), "audio/wav")}
        try:
            response = await self._client.post(
                self._url, data=data, files=files, headers=_headers(self._key)
            )
            if not response.is_success:
                raise SpeechError(f"STT endpoint returned HTTP {response.status_code}")
        except httpx.HTTPError:
            raise SpeechError("STT endpoint transport failure") from None
        try:
            payload: Any = response.json()
            text = payload["text"]
        except (ValueError, TypeError, KeyError):
            raise SpeechError("STT endpoint returned an invalid response") from None
        if not isinstance(text, str):
            raise SpeechError("STT endpoint returned an invalid response")
        return text[:MAX_TRANSCRIPT_CHARS]

    async def close(self) -> None:
        await self._client.aclose()


class OpenAISynthesizer:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.model = settings.tts_model
        self.voice = settings.tts_voice
        self.format = settings.tts_format
        self.speed = settings.tts_speed
        self.pcm_sample_rate = settings.tts_pcm_sample_rate
        self._url = _endpoint(settings.tts_base_url, "audio/speech")
        self._key = settings.tts_api_key.get_secret_value() if settings.tts_api_key else ""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.tts_timeout_seconds, connect=5.0),
            follow_redirects=False,
            transport=transport,
        )

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        body = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "response_format": self.format,
            "speed": self.speed,
        }
        parser = WavStreamParser() if self.format == "wav" else None
        carry = b""
        try:
            async with self._client.stream(
                "POST", self._url, json=body, headers=_headers(self._key)
            ) as response:
                if not response.is_success:
                    raise SpeechError(f"TTS endpoint returned HTTP {response.status_code}")
                async for chunk in response.aiter_bytes():
                    if parser is not None:
                        try:
                            pcm = parser.feed(chunk)
                        except ValueError:
                            raise SpeechError("TTS endpoint returned invalid WAV audio") from None
                        rate = parser.sample_rate
                    else:
                        data = carry + chunk
                        cut = len(data) - (len(data) % 2)
                        pcm, carry = data[:cut], data[cut:]
                        rate = self.pcm_sample_rate
                    if rate is not None and not MIN_RATE <= rate <= MAX_RATE:
                        raise SpeechError("TTS endpoint returned an unsupported sample rate")
                    if pcm and rate:
                        yield AudioChunk(pcm, rate)
        except httpx.HTTPError:
            raise SpeechError("TTS endpoint transport failure") from None
        if parser is not None and parser.sample_rate is None:
            raise SpeechError("TTS endpoint returned invalid WAV audio")

    async def close(self) -> None:
        await self._client.aclose()
