"""WebSocket transports for the voice gateway.

* ``/v1/voice/twilio`` - Twilio Media Streams (bidirectional ``<Connect><Stream>``):
  8 kHz mu-law in/out, 20 ms ``media`` frames, ``mark`` for playback acknowledgement and
  ``clear`` for barge-in.
* ``/v1/voice/stream`` - generic bridge/benchmark protocol: binary PCM16 16 kHz frames in,
  JSON events plus binary PCM16 16 kHz audio out.
"""

import asyncio
import base64
import binascii
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status
from prometheus_client import CollectorRegistry

from ledger_rag.config import Settings
from ledger_rag.models import Principal
from ledger_rag.voice.codec import mulaw_decode, mulaw_encode
from ledger_rag.voice.gateway import (
    AnswerEvent,
    AudioEvent,
    ClearEvent,
    ErrorEvent,
    MarkEvent,
    TranscriptEvent,
    TurnMetricsEvent,
    VoiceDependencies,
    VoiceEvent,
    VoiceSession,
)
from ledger_rag.voice.metrics import VoiceMetrics
from ledger_rag.voice.speech import (
    OpenAISynthesizer,
    OpenAITranscriber,
    SpeechToText,
    TextToSpeech,
)

if TYPE_CHECKING:
    from ledger_rag.api import Runtime

logger = structlog.get_logger()
router = APIRouter()

TWILIO_RATE = 8000
TWILIO_FRAME_BYTES = 160  # 20 ms of 8 kHz mu-law
STREAM_RATE = 16000
AUTH_DEADLINE_SECONDS = 10.0
MAX_PRE_START_MESSAGES = 4
SEND_TIMEOUT_SECONDS = 5.0


@dataclass
class VoiceRuntime:
    stt: SpeechToText
    tts: TextToSpeech
    metrics: VoiceMetrics
    sessions: asyncio.Semaphore

    async def close(self) -> None:
        await self.stt.close()
        await self.tts.close()


def make_voice_runtime(
    settings: Settings,
    registry: CollectorRegistry,
    stt: SpeechToText | None = None,
    tts: TextToSpeech | None = None,
) -> VoiceRuntime:
    return VoiceRuntime(
        stt=stt or OpenAITranscriber(settings),
        tts=tts or OpenAISynthesizer(settings),
        metrics=VoiceMetrics(registry),
        sessions=asyncio.Semaphore(settings.max_voice_sessions),
    )


class _Reject(Exception):
    def __init__(self, code: int, reason: str, outcome: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.outcome = outcome


async def _close(websocket: WebSocket, code: int, reason: str = "") -> None:
    with suppress(RuntimeError, WebSocketDisconnect, OSError):
        await websocket.close(code=code, reason=reason)


async def _authenticate(state: "Runtime", authorization: str) -> Principal:
    try:
        principal = await state.auth.authenticate(authorization)
        if "reader" not in principal.roles:
            raise _Reject(status.WS_1008_POLICY_VIOLATION, "reader role required", "forbidden")
        state.limiter.check(principal)
    except HTTPException as exc:
        if exc.status_code == 429 or exc.status_code >= 500:
            raise _Reject(
                status.WS_1013_TRY_AGAIN_LATER, "Service unavailable", "unavailable"
            ) from None
        raise _Reject(
            status.WS_1008_POLICY_VIOLATION, "Authentication failed", "unauthorized"
        ) from None
    return principal


async def _receive(websocket: WebSocket, limit: int) -> str | bytes | None:
    message = await websocket.receive()
    if message["type"] == "websocket.disconnect":
        return None
    payload: str | bytes | None = message.get("text")
    if payload is None:
        payload = message.get("bytes")
    if payload is None:
        return None
    if len(payload) > limit:
        raise _Reject(status.WS_1009_MESSAGE_TOO_BIG, "Frame too large", "frame_too_large")
    return payload


def _json(payload: str | bytes) -> dict[str, Any]:
    try:
        data = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        data = None
    if not isinstance(data, dict):
        raise _Reject(
            status.WS_1007_INVALID_FRAME_PAYLOAD_DATA, "Invalid JSON message", "protocol_error"
        )
    return data


@asynccontextmanager
async def _session_slot(voice: VoiceRuntime) -> AsyncIterator[None]:
    try:
        await asyncio.wait_for(voice.sessions.acquire(), timeout=0.01)
    except TimeoutError:
        raise _Reject(
            status.WS_1013_TRY_AGAIN_LATER, "Voice capacity exhausted", "at_capacity"
        ) from None
    voice.metrics.active_sessions.inc()
    try:
        yield
    finally:
        voice.metrics.active_sessions.dec()
        voice.sessions.release()


def _dependencies(state: "Runtime", voice: VoiceRuntime) -> VoiceDependencies:
    return VoiceDependencies(
        settings=state.settings,
        service=state.service,
        stt=voice.stt,
        tts=voice.tts,
        metrics=voice.metrics,
        rag_metrics=state.metrics,
        rag_slots=state.slots,
        limiter=state.limiter,
    )


class _Sender:
    """Serializes sends from the receive loop and the turn task; bounds slow consumers."""

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.lock = asyncio.Lock()

    async def text(self, payload: dict[str, Any]) -> None:
        async with self.lock, asyncio.timeout(SEND_TIMEOUT_SECONDS):
            await self.websocket.send_text(json.dumps(payload))

    async def binary(self, payload: bytes) -> None:
        async with self.lock, asyncio.timeout(SEND_TIMEOUT_SECONDS):
            await self.websocket.send_bytes(payload)


class TwilioSink:
    def __init__(self, sender: _Sender, stream_sid: str) -> None:
        self.sender = sender
        self.stream_sid = stream_sid
        self.buffer = bytearray()

    async def __call__(self, event: VoiceEvent) -> None:
        if isinstance(event, AudioEvent):
            self.buffer.extend(mulaw_encode(event.pcm))
            while len(self.buffer) >= TWILIO_FRAME_BYTES:
                frame = bytes(self.buffer[:TWILIO_FRAME_BYTES])
                del self.buffer[:TWILIO_FRAME_BYTES]
                await self._media(frame)
        elif isinstance(event, MarkEvent):
            if self.buffer:
                padding = b"\xff" * (TWILIO_FRAME_BYTES - len(self.buffer))
                frame = bytes(self.buffer) + padding
                self.buffer.clear()
                await self._media(frame)
            await self.sender.text(
                {"event": "mark", "streamSid": self.stream_sid, "mark": {"name": event.name}}
            )
        elif isinstance(event, ClearEvent):
            self.buffer.clear()
            await self.sender.text({"event": "clear", "streamSid": self.stream_sid})

    async def _media(self, frame: bytes) -> None:
        await self.sender.text(
            {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": base64.b64encode(frame).decode()},
            }
        )


class StreamSink:
    def __init__(self, sender: _Sender) -> None:
        self.sender = sender

    async def __call__(self, event: VoiceEvent) -> None:
        if isinstance(event, AudioEvent):
            await self.sender.binary(event.pcm)
        elif isinstance(event, TranscriptEvent):
            await self.sender.text(
                {"type": "transcript", "turn_id": event.turn_id, "text": event.text}
            )
        elif isinstance(event, AnswerEvent):
            answer = event.answer
            await self.sender.text(
                {
                    "type": "answer",
                    "turn_id": event.turn_id,
                    "text": event.speakable,
                    "abstained": answer.abstained if answer else True,
                    "request_id": answer.request_id if answer else None,
                    "citations": (
                        [citation.model_dump(mode="json") for citation in answer.citations]
                        if answer
                        else []
                    ),
                }
            )
        elif isinstance(event, MarkEvent):
            await self.sender.text({"type": "audio_end", "turn_id": event.name})
        elif isinstance(event, ClearEvent):
            await self.sender.text({"type": "clear", "turn_id": event.turn_id})
        elif isinstance(event, TurnMetricsEvent):
            await self.sender.text({"type": "turn_metrics", **event.metrics.as_dict()})
        elif isinstance(event, ErrorEvent):
            await self.sender.text({"type": "error", "detail": event.detail})


def _voice(websocket: WebSocket) -> tuple["Runtime", VoiceRuntime | None]:
    state: Runtime = websocket.app.state.runtime
    return state, state.voice


async def _twilio_start(
    websocket: WebSocket, state: "Runtime", principal: Principal | None
) -> tuple[str, Principal]:
    try:
        async with asyncio.timeout(AUTH_DEADLINE_SECONDS):
            return await _twilio_handshake(websocket, state, principal)
    except TimeoutError:
        raise _Reject(
            status.WS_1008_POLICY_VIOLATION, "Authentication timeout", "unauthorized"
        ) from None


async def _twilio_handshake(
    websocket: WebSocket, state: "Runtime", principal: Principal | None
) -> tuple[str, Principal]:
    limit = state.settings.voice_max_frame_bytes
    for _ in range(MAX_PRE_START_MESSAGES):
        payload = await _receive(websocket, limit)
        if payload is None:
            raise _Reject(status.WS_1000_NORMAL_CLOSURE, "", "disconnected")
        message = _json(payload)
        event = message.get("event")
        if event == "connected":
            continue
        if event != "start":
            break
        start = message.get("start")
        if not isinstance(start, dict):
            break
        stream_sid = start.get("streamSid") or message.get("streamSid")
        media_format = start.get("mediaFormat") or {}
        if (
            not isinstance(stream_sid, str)
            or not 0 < len(stream_sid) <= 128
            or not isinstance(media_format, dict)
            or media_format.get("encoding", "audio/x-mulaw") != "audio/x-mulaw"
            or media_format.get("sampleRate", TWILIO_RATE) != TWILIO_RATE
            or media_format.get("channels", 1) != 1
        ):
            raise _Reject(status.WS_1003_UNSUPPORTED_DATA, "Unsupported stream", "protocol_error")
        if principal is None:
            parameters = start.get("customParameters") or {}
            token = parameters.get("token") if isinstance(parameters, dict) else None
            if not isinstance(token, str) or not token:
                raise _Reject(
                    status.WS_1008_POLICY_VIOLATION, "Authentication required", "unauthorized"
                )
            principal = await _authenticate(state, f"Bearer {token}")
        return stream_sid, principal
    raise _Reject(status.WS_1008_POLICY_VIOLATION, "start event required", "unauthorized")


@router.websocket("/v1/voice/twilio")
async def twilio_media_stream(websocket: WebSocket) -> None:
    """Twilio Media Streams endpoint.

    Twilio cannot attach arbitrary headers to the stream, so the bearer token may be
    passed as ``<Parameter name="token">`` (``start.customParameters.token``). In
    production that value must be a short-lived, single-call token minted by the
    authenticated TwiML webhook (validate ``X-Twilio-Signature`` there), never a
    long-lived user credential. No media is processed before authentication.
    """
    state, voice = _voice(websocket)
    if voice is None:
        await _close(websocket, status.WS_1008_POLICY_VIOLATION, "Voice gateway disabled")
        return
    outcome = "completed"
    try:
        # Browsers always send Origin; Twilio does not. Blocks cross-site WebSocket hijacking.
        if websocket.headers.get("origin"):
            raise _Reject(status.WS_1008_POLICY_VIOLATION, "Browser origin rejected", "forbidden")
        header = websocket.headers.get("authorization")
        principal = await _authenticate(state, header) if header else None
        await websocket.accept()
        stream_sid, principal = await _twilio_start(websocket, state, principal)
        async with _session_slot(voice):
            sender = _Sender(websocket)
            session = VoiceSession(
                _dependencies(state, voice),
                principal,
                TwilioSink(sender, stream_sid),
                input_rate=TWILIO_RATE,
                output_rate=TWILIO_RATE,
                await_playback_ack=True,
                transport="twilio",
                session_id=uuid.uuid4().hex,
            )
            logger.info("voice_session_started", session_id=session.session_id, transport="twilio")
            try:
                outcome = await _twilio_loop(websocket, state, session)
            finally:
                await session.close()
            if outcome == "timeout":
                await _close(websocket, status.WS_1000_NORMAL_CLOSURE, "Session limit reached")
            elif outcome == "completed":
                await _close(websocket, status.WS_1000_NORMAL_CLOSURE)
    except _Reject as reject:
        outcome = reject.outcome
        await _close(websocket, reject.code, reject.reason)
    except (WebSocketDisconnect, TimeoutError):
        outcome = "disconnected"
    finally:
        voice.metrics.sessions.labels("twilio", outcome).inc()
        logger.info("voice_session_ended", transport="twilio", outcome=outcome)


async def _twilio_loop(websocket: WebSocket, state: "Runtime", session: VoiceSession) -> str:
    limit = state.settings.voice_max_frame_bytes
    try:
        async with asyncio.timeout(state.settings.voice_max_session_seconds):
            while True:
                payload = await _receive(websocket, limit)
                if payload is None:
                    return "disconnected"
                message = _json(payload)
                event = message.get("event")
                if event == "media":
                    media = message.get("media")
                    if not isinstance(media, dict) or media.get("track", "inbound") != "inbound":
                        continue
                    encoded = media.get("payload")
                    try:
                        audio = base64.b64decode(encoded, validate=True)  # type: ignore[arg-type]
                    except (binascii.Error, TypeError, ValueError):
                        raise _Reject(
                            status.WS_1007_INVALID_FRAME_PAYLOAD_DATA,
                            "Invalid media payload",
                            "protocol_error",
                        ) from None
                    await session.feed(mulaw_decode(audio))
                elif event == "mark":
                    mark = message.get("mark")
                    if isinstance(mark, dict) and isinstance(mark.get("name"), str):
                        await session.playback_complete(mark["name"])
                elif event == "stop":
                    return "completed"
    except TimeoutError:
        return "timeout"


@router.websocket("/v1/voice/stream")
async def voice_stream(websocket: WebSocket) -> None:
    """Generic PCM16/16 kHz bridge. Requires ``Authorization: Bearer`` on the handshake."""
    state, voice = _voice(websocket)
    if voice is None:
        await _close(websocket, status.WS_1008_POLICY_VIOLATION, "Voice gateway disabled")
        return
    outcome = "completed"
    try:
        header = websocket.headers.get("authorization")
        if not header:
            raise _Reject(
                status.WS_1008_POLICY_VIOLATION, "Authentication required", "unauthorized"
            )
        principal = await _authenticate(state, header)
        async with _session_slot(voice):
            await websocket.accept()
            sender = _Sender(websocket)
            session = VoiceSession(
                _dependencies(state, voice),
                principal,
                StreamSink(sender),
                input_rate=STREAM_RATE,
                output_rate=STREAM_RATE,
                await_playback_ack=False,
                transport="stream",
            )
            await sender.text(
                {"type": "ready", "session_id": session.session_id, "sample_rate": STREAM_RATE}
            )
            try:
                outcome = await _stream_loop(websocket, state, session, sender)
            finally:
                await session.close()
            if outcome == "timeout":
                await _close(websocket, status.WS_1000_NORMAL_CLOSURE, "Session limit reached")
    except _Reject as reject:
        outcome = reject.outcome
        await _close(websocket, reject.code, reject.reason)
    except (WebSocketDisconnect, TimeoutError):
        outcome = "disconnected"
    finally:
        voice.metrics.sessions.labels("stream", outcome).inc()
        logger.info("voice_session_ended", transport="stream", outcome=outcome)


async def _stream_loop(
    websocket: WebSocket, state: "Runtime", session: VoiceSession, sender: _Sender
) -> str:
    limit = state.settings.voice_max_frame_bytes
    try:
        async with asyncio.timeout(state.settings.voice_max_session_seconds):
            while True:
                payload = await _receive(websocket, limit)
                if payload is None:
                    return "disconnected"
                if isinstance(payload, bytes):
                    await session.feed(payload)
                    continue
                kind = _json(payload).get("type")
                if kind == "end":
                    await session.end_utterance()
                elif kind != "ping":
                    await sender.text({"type": "error", "detail": "Unknown control message"})
    except TimeoutError:
        return "timeout"
