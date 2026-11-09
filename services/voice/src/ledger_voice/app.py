"""``ledger-voice`` HTTP/WebSocket service (ADR 0006).

    POST /twilio/voice   Twilio webhook: verify X-Twilio-Signature and the dialled
                         line, mint a per-call stream token, return <Connect><Stream>
    WS   /twilio/media   Twilio Media Streams: verify the token before any media,
                         then run one Pipecat pipeline for the call
    GET  /health/live, /health/ready

The service holds only speech, Twilio and engine-client credentials. Answers come
from the engine's public API (``profile=voice``).
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import structlog
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.websockets import WebSocketDisconnect, WebSocketState
from twilio.request_validator import RequestValidator
from twilio.twiml.voice_response import Connect, VoiceResponse

from ledger_voice.config import VoiceSettings
from ledger_voice.engine import EngineClient
from ledger_voice.logs import configure_logging
from ledger_voice.tokens import TokenVerifier, mint

logger = structlog.get_logger()
START_TIMEOUT_SECONDS = 5.0
MAX_PRESTART_MESSAGES = 3
MEDIA_PATH = "/twilio/media"


@dataclass(frozen=True)
class StreamStart:
    stream_sid: str
    call_sid: str
    account_sid: str


SessionRunner = Callable[[WebSocket, StreamStart, "VoiceRuntime"], Awaitable[None]]


@dataclass
class VoiceRuntime:
    settings: VoiceSettings
    engine: EngineClient
    verifier: TokenVerifier | None
    sessions: asyncio.Semaphore
    run_session: SessionRunner


def _public_ws_url(public_url: str) -> str:
    parts = urlsplit(public_url.rstrip("/"))
    scheme = "wss" if parts.scheme == "https" else "ws"
    return f"{scheme}://{parts.netloc}{parts.path}{MEDIA_PATH}"


async def run_pipecat_session(
    websocket: WebSocket, start: StreamStart, runtime: VoiceRuntime
) -> None:
    """One call: Twilio serializer + FastAPI transport around the Ledger pipeline."""
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.serializers.twilio import TwilioFrameSerializer
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )
    from pipecat.workers.runner import WorkerRunner

    from ledger_voice.answer import EngineAnswerProcessor
    from ledger_voice.pipeline import TELEPHONY_SAMPLE_RATE, build_pipeline, speech_services

    settings = runtime.settings
    token = settings.twilio_auth_token.get_secret_value() if settings.twilio_auth_token else None
    # Hang up through the REST API only for the configured account (never for simulators).
    hang_up = bool(token and settings.twilio_account_sid == start.account_sid)
    serializer = TwilioFrameSerializer(
        stream_sid=start.stream_sid,
        call_sid=start.call_sid,
        account_sid=start.account_sid if hang_up else None,
        auth_token=token if hang_up else None,
        params=TwilioFrameSerializer.InputParams(auto_hang_up=hang_up),
    )
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
            session_timeout=settings.max_call_seconds,
        ),
    )
    turn = 0

    async def on_reply(transcript: str, reply: Any) -> None:
        nonlocal turn
        turn += 1
        # Counts and timings only: no transcript or answer text in logs.
        logger.info(
            "voice_turn",
            call=start.call_sid[-6:],
            turn=turn,
            outcome=reply.outcome,
            engine_ms=reply.engine_ms,
            transcript_chars=len(transcript),
            spoken_chars=len(reply.text),
        )

    stt, tts = speech_services(settings)
    answer = EngineAnswerProcessor(
        runtime.engine, max_transcript_chars=settings.max_transcript_chars, on_reply=on_reply
    )
    pipeline = build_pipeline(transport.input(), transport.output(), answer, stt=stt, tts=tts)
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=TELEPHONY_SAMPLE_RATE,
            audio_out_sample_rate=TELEPHONY_SAMPLE_RATE,
            enable_metrics=True,
        ),
        idle_timeout_secs=120,
    )

    async def on_disconnected(_transport: Any, _websocket: Any) -> None:
        await worker.cancel(reason="caller disconnected")

    transport.add_event_handler("on_client_disconnected", on_disconnected)

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    async with asyncio.timeout(settings.max_call_seconds + 30):
        await runner.run()


def create_app(
    settings: VoiceSettings | None = None,
    *,
    run_session: SessionRunner = run_pipecat_session,
    engine: EngineClient | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        resolved = settings or VoiceSettings()
        client = engine or EngineClient(resolved)
        secret = resolved.stream_secret.get_secret_value() if resolved.stream_secret else None
        app.state.runtime = VoiceRuntime(
            settings=resolved,
            engine=client,
            verifier=TokenVerifier(secret, resolved.stream_token_seconds) if secret else None,
            sessions=asyncio.Semaphore(resolved.max_calls),
            run_session=run_session,
        )
        try:
            yield
        finally:
            await client.close()

    app = FastAPI(title="ledger-voice", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready(request: Request) -> Response:
        runtime: VoiceRuntime = request.app.state.runtime
        try:
            runtime.settings.require_live()
        except RuntimeError:
            return Response('{"status":"not_configured"}', 503, media_type="application/json")
        return Response('{"status":"ready"}', 200, media_type="application/json")

    @app.post("/twilio/voice")
    async def twilio_voice(request: Request) -> Response:
        runtime: VoiceRuntime = request.app.state.runtime
        settings = runtime.settings
        if (
            settings.twilio_auth_token is None
            or runtime.verifier is None
            or not settings.public_url
        ):
            raise HTTPException(503, "Twilio is not configured")
        body = await request.body()
        if len(body) > 16_384:
            raise HTTPException(413, "Request too large")
        form = await request.form()
        params = {key: str(value) for key, value in form.items()}
        url = settings.public_url.rstrip("/") + "/twilio/voice"
        signature = request.headers.get("X-Twilio-Signature", "")
        validator = RequestValidator(settings.twilio_auth_token.get_secret_value())
        if not signature or not validator.validate(url, params, signature):
            logger.warning("twilio_signature_rejected")
            raise HTTPException(403, "Invalid signature")
        call_sid = params.get("CallSid", "")
        if not call_sid.startswith("CA") or len(call_sid) > 64:
            raise HTTPException(400, "Missing call")
        response = VoiceResponse()
        if params.get("To") not in settings.allowed_numbers:
            logger.warning("twilio_line_rejected")
            response.reject()
            return Response(str(response), media_type="application/xml")
        assert settings.stream_secret is not None
        token = mint(
            settings.stream_secret.get_secret_value(), call_sid, settings.stream_token_seconds
        )
        connect = Connect()
        stream = connect.stream(url=_public_ws_url(settings.public_url))
        stream.parameter(name="token", value=token)
        response.append(connect)
        return Response(str(response), media_type="application/xml")

    @app.websocket(MEDIA_PATH)
    async def twilio_media(websocket: WebSocket) -> None:
        runtime: VoiceRuntime = websocket.app.state.runtime
        if websocket.headers.get("origin"):
            await websocket.close(code=1008)  # browsers never speak Twilio Media Streams
            return
        if runtime.verifier is None:
            await websocket.close(code=1011)
            return
        await websocket.accept()
        try:
            start = await _await_start(websocket, runtime)
        except (TimeoutError, ValueError, WebSocketDisconnect):
            start = None
        if start is None:
            logger.warning("voice_stream_rejected")
            await _close(websocket, 1008)
            return
        if runtime.sessions.locked():
            logger.warning("voice_capacity_exhausted")
            await _close(websocket, 1013)
            return
        async with runtime.sessions:
            started = time.monotonic()
            try:
                await runtime.run_session(websocket, start, runtime)
            except Exception as exc:  # sanitized: class only, never payloads
                logger.error("voice_session_failed", error_class=type(exc).__name__)
            finally:
                logger.info("voice_session_closed", seconds=round(time.monotonic() - started, 1))
                await _close(websocket, 1000)

    return app


async def _await_start(websocket: WebSocket, runtime: VoiceRuntime) -> StreamStart | None:
    """Read Twilio's ``connected``/``start`` messages; verify the token before media."""
    limit = runtime.settings.max_message_bytes
    async with asyncio.timeout(START_TIMEOUT_SECONDS):
        for _ in range(MAX_PRESTART_MESSAGES):
            raw = await websocket.receive_text()
            if len(raw) > limit:
                return None
            message = json.loads(raw)
            if not isinstance(message, dict):
                return None
            event = message.get("event")
            if event == "connected":
                continue
            if event != "start" or not isinstance(message.get("start"), dict):
                return None
            data = message["start"]
            parameters = data.get("customParameters") or {}
            stream_sid, call_sid = data.get("streamSid"), data.get("callSid")
            account_sid = data.get("accountSid", "")
            if not all(isinstance(v, str) and 0 < len(v) <= 64 for v in (stream_sid, call_sid)):
                return None
            if not isinstance(parameters, dict) or not isinstance(account_sid, str):
                return None
            assert runtime.verifier is not None
            if not runtime.verifier.verify(parameters.get("token"), call_sid):
                return None
            return StreamStart(stream_sid, call_sid, account_sid[:64])
    return None


async def _close(websocket: WebSocket, code: int) -> None:
    if websocket.client_state != WebSocketState.DISCONNECTED:
        try:
            await websocket.close(code=code)
        except RuntimeError:
            pass


def main() -> None:
    import uvicorn

    configure_logging()
    settings = VoiceSettings()
    uvicorn.run(
        create_app(settings),
        host="127.0.0.1",
        port=8090,
        ws_max_size=settings.max_message_bytes,
        access_log=False,
    )
