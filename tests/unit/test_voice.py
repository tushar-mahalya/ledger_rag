import asyncio
import base64
import json
import math
import struct
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.websockets import WebSocketDisconnect

from ledger_rag.api.app import create_app
from ledger_rag.core.config import Settings
from ledger_rag.core.models import Answer, Document, Principal, Profile
from ledger_rag.generation.providers import make_provider
from ledger_rag.ingestion.chunking import chunk_document
from ledger_rag.orchestration.service import ABSTENTION, RagService
from ledger_rag.retrieval.memory import MemoryRetriever
from ledger_rag.voice.codec import (
    Resampler,
    WavStreamParser,
    build_wav,
    mulaw_decode,
    mulaw_encode,
    parse_wav,
    pcm16_to_samples,
    resample,
    samples_to_pcm16,
)
from ledger_rag.voice.gateway import (
    ABSTENTION_PROMPT,
    AnswerEvent,
    AudioEvent,
    ClearEvent,
    MarkEvent,
    TranscriptEvent,
    TurnMetricsEvent,
    VoiceDependencies,
    VoiceEvent,
    VoiceSession,
    VoiceState,
    is_meaningful,
    to_speakable,
)
from ledger_rag.voice.metrics import VoiceMetrics
from ledger_rag.voice.speech import (
    AudioChunk,
    OpenAISynthesizer,
    OpenAITranscriber,
    SpeechError,
)
from ledger_rag.voice.vad import EnergyVad, VadConfig, VadEventKind

TOKEN = "test-only-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
QUESTION = "What was revenue in 2024?"


def tone(rate: int, ms: int, amplitude: int = 8000, freq: float = 440.0) -> bytes:
    count = rate * ms // 1000
    return samples_to_pcm16(
        [int(amplitude * math.sin(2 * math.pi * freq * i / rate)) for i in range(count)]
    )


def silence(rate: int, ms: int) -> bytes:
    return b"\x00\x00" * (rate * ms // 1000)


# --- codec -----------------------------------------------------------------------------


def test_mulaw_known_values_and_round_trip() -> None:
    pcm = samples_to_pcm16([0, 1, -1, 32767, -32768, 1000, -1000])
    assert mulaw_encode(pcm) == bytes([0xFF, 0xFF, 0x7F, 0x80, 0x00, 0xCE, 0x4E])
    assert list(pcm16_to_samples(mulaw_decode(bytes([0xFF, 0x7F, 0x80, 0x00])))) == [
        0,
        0,
        32124,
        -32124,
    ]
    signal = tone(8000, 100)
    decoded = pcm16_to_samples(mulaw_decode(mulaw_encode(signal)))
    original = pcm16_to_samples(signal)
    assert len(decoded) == len(original)
    # G.711 is logarithmic: quantization error is bounded relative to magnitude.
    assert all(abs(a - b) <= max(8, abs(b) // 16) for a, b in zip(decoded, original, strict=True))
    assert mulaw_decode(mulaw_encode(mulaw_decode(bytes(range(256))))) == mulaw_decode(
        bytes(range(256))
    )


@pytest.mark.parametrize(
    ("source", "target", "samples", "expected"),
    [
        (8000, 16000, 160, 320),
        (16000, 8000, 320, 160),
        (24000, 8000, 480, 160),
        (24000, 16000, 480, 320),
        (8000, 24000, 160, 480),
    ],
)
def test_resample_lengths(source: int, target: int, samples: int, expected: int) -> None:
    pcm = tone(source, samples * 1000 // source)
    assert len(resample(pcm, source, target)) == expected * 2
    streaming = Resampler(source, target)
    out = b"".join(streaming.process(pcm[i : i + 77]) for i in range(0, len(pcm), 77))
    assert abs(len(out) // 2 - expected) <= 3


def test_resample_preserves_tone_amplitude_and_rejects_bad_rates() -> None:
    out = pcm16_to_samples(resample(tone(24000, 100, 10000, 300), 24000, 8000))
    assert 9000 < max(out) <= 10000
    with pytest.raises(ValueError):
        resample(b"\x00\x00", 4000, 8000)


def test_wav_build_parse_and_stream_parser() -> None:
    pcm = tone(16000, 50)
    wav = build_wav(pcm, 16000)
    assert parse_wav(wav) == (pcm, 16000)
    parser = WavStreamParser()
    out = b"".join(parser.feed(wav[i : i + 7]) for i in range(0, len(wav), 7))
    assert parser.sample_rate == 16000 and out == pcm
    with pytest.raises(ValueError):
        parse_wav(b"not a wav")
    with pytest.raises(ValueError):
        WavStreamParser().feed(b"RIFX\x00\x00\x00\x00WAVEjunk")


def test_stream_parser_handles_extra_chunks_stereo_and_unknown_size() -> None:
    stereo = samples_to_pcm16([100, 300, -100, -300])
    fmt = struct.pack("<HHIIHH", 1, 2, 24000, 24000 * 4, 4, 16)
    header = (
        b"RIFF\xff\xff\xff\xffWAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"LIST\x03\x00\x00\x00abc\x00"
        + b"data\xff\xff\xff\xff"
    )
    parser = WavStreamParser()
    assert parser.feed(header + stereo[:5]) == samples_to_pcm16([200])
    assert parser.feed(stereo[5:]) == samples_to_pcm16([-200])
    assert parser.sample_rate == 24000


# --- VAD -------------------------------------------------------------------------------


def vad_config(**overrides: Any) -> VadConfig:
    return VadConfig(**{"max_utterance_seconds": 2.0, **overrides})


def test_vad_endpointing() -> None:
    vad = EnergyVad(vad_config(), 8000)
    assert vad.process(silence(8000, 400)) == []
    events = vad.process(tone(8000, 400))
    assert [event.kind for event in events] == [VadEventKind.SPEECH_START]
    assert vad.process(silence(8000, 300)) == []
    events = vad.process(silence(8000, 300))
    assert [event.kind for event in events] == [VadEventKind.SPEECH_END]
    duration_ms = len(events[0].audio) / 2 / 8000 * 1000
    # pre-roll (<=200 ms) + 400 ms speech + ~100 ms kept tail
    assert 400 <= duration_ms <= 720
    assert not vad.in_utterance


def test_vad_ignores_blips_and_caps_long_utterances() -> None:
    vad = EnergyVad(vad_config(), 16000)
    assert vad.process(tone(16000, 100) + silence(16000, 600)) == []
    vad = EnergyVad(vad_config(max_utterance_seconds=1.0), 16000)
    events = vad.process(tone(16000, 1500))
    # The cap forces an endpoint; continued speech then starts a new utterance.
    assert [event.kind for event in events] == [
        VadEventKind.SPEECH_START,
        VadEventKind.SPEECH_END,
        VadEventKind.SPEECH_START,
    ]
    assert events[1].capped and len(events[1].audio) == 16000 * 2


def test_vad_barge_in_and_flush() -> None:
    vad = EnergyVad(vad_config(min_speech_ms=100, barge_in_ms=300), 8000)
    assert vad.process(tone(8000, 200), playing=True) == []
    assert [e.kind for e in vad.process(tone(8000, 100), playing=True)] == [VadEventKind.BARGE_IN]
    flushed = vad.flush()
    assert flushed is not None and flushed.kind is VadEventKind.SPEECH_END
    assert vad.flush() is None
    quiet = EnergyVad(vad_config(speech_rms=20000), 8000)
    assert quiet.process(tone(8000, 500)) == []


# --- speech adapters -------------------------------------------------------------------


def speech_settings(**overrides: Any) -> Settings:
    return Settings(
        demo_token=SecretStr(TOKEN),
        stt_base_url="https://stt.example/v1",
        stt_api_key=SecretStr("stt-secret"),
        tts_base_url="https://tts.example/v1/",
        tts_api_key=SecretStr("tts-secret"),
        **overrides,
    )


async def test_stt_multipart_request_and_parsing() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"text": " What was revenue? "})

    stt = OpenAITranscriber(speech_settings(), transport=httpx.MockTransport(handler))
    assert await stt.transcribe(tone(16000, 100), 16000) == " What was revenue? "
    await stt.close()
    request = seen[0]
    body = request.content
    assert str(request.url) == "https://stt.example/v1/audio/transcriptions"
    assert request.headers["authorization"] == "Bearer stt-secret"
    assert request.headers["content-type"].startswith("multipart/form-data")
    assert b'name="model"\r\n\r\nwhisper-1' in body
    assert b'name="language"\r\n\r\nen' in body
    assert b'name="response_format"\r\n\r\njson' in body
    assert b'filename="utterance.wav"' in body and b"RIFF" in body and b"WAVEfmt" in body


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(500, text="secret upstream detail stt-secret"), "HTTP 500"),
        (httpx.Response(302, headers={"location": "https://evil.example"}), "HTTP 302"),
        (httpx.Response(200, text="not json"), "invalid response"),
        (httpx.Response(200, json={"text": 5}), "invalid response"),
    ],
)
async def test_stt_errors_are_sanitized(response: httpx.Response, message: str) -> None:
    stt = OpenAITranscriber(
        speech_settings(), transport=httpx.MockTransport(lambda request: response)
    )
    with pytest.raises(SpeechError) as error:
        await stt.transcribe(b"\x00\x00", 8000)
    assert message in str(error.value)
    assert "secret" not in str(error.value) and error.value.__cause__ is None
    await stt.close()


async def test_stt_transport_failure_is_sanitized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connect to https://stt.example?key=stt-secret failed")

    stt = OpenAITranscriber(speech_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(SpeechError, match="transport failure") as error:
        await stt.transcribe(b"\x00\x00", 8000)
    assert "secret" not in str(error.value)
    await stt.close()


async def test_tts_streams_pcm_before_synthesis_finishes() -> None:
    release = asyncio.Event()
    seen: list[httpx.Request] = []

    async def body() -> AsyncIterator[bytes]:
        yield b"\x01\x00\x02"  # odd split across chunk boundary
        await release.wait()
        yield b"\x00\x03\x00"

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=body())

    tts = OpenAISynthesizer(speech_settings(), transport=httpx.MockTransport(handler))
    chunks: list[AudioChunk] = []
    async for chunk in tts.synthesize("Hello there."):
        chunks.append(chunk)
        release.set()  # only reachable if the first chunk arrived before the stream ended
    await tts.close()
    assert [c.pcm for c in chunks] == [b"\x01\x00", b"\x02\x00\x03\x00"]
    assert {c.sample_rate for c in chunks} == {24000}
    request = seen[0]
    assert str(request.url) == "https://tts.example/v1/audio/speech"
    assert request.headers["authorization"] == "Bearer tts-secret"
    assert json.loads(request.content) == {
        "model": "gpt-4o-mini-tts",
        "voice": "alloy",
        "input": "Hello there.",
        "response_format": "pcm",
        "speed": 1.0,
    }


async def test_tts_wav_stream_and_errors() -> None:
    wav = build_wav(tone(16000, 40), 16000)

    async def body() -> AsyncIterator[bytes]:
        for index in range(0, len(wav), 11):
            yield wav[index : index + 11]

    tts = OpenAISynthesizer(
        speech_settings(tts_format="wav"),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body())),
    )
    chunks = [chunk async for chunk in tts.synthesize("Hi")]
    assert b"".join(c.pcm for c in chunks) == tone(16000, 40)
    assert {c.sample_rate for c in chunks} == {16000}
    await tts.close()

    for response, message in [
        (httpx.Response(401, text="bad key tts-secret"), "HTTP 401"),
        (httpx.Response(200, content=b"<html>nope</html>"), "invalid WAV"),
    ]:
        failing = OpenAISynthesizer(
            speech_settings(tts_format="wav"),
            transport=httpx.MockTransport(lambda request, response=response: response),
        )
        with pytest.raises(SpeechError, match=message) as error:
            [chunk async for chunk in failing.synthesize("Hi")]
        assert "secret" not in str(error.value)
        await failing.close()


def test_production_requires_https_speech_endpoints() -> None:
    base: dict[str, Any] = {
        "environment": "production",
        "auth_mode": "oidc",
        "oidc_issuer": "https://issuer.example",
        "oidc_audience": "ledger",
        "oidc_jwks_url": "https://issuer.example/jwks",
        "retrieval_backend": "qdrant",
        "embedding_api_key": SecretStr("k"),
        "provider": "openai",
        "llm_api_key": SecretStr("k"),
        "llm_model": "m",
    }
    Settings(**base)
    with pytest.raises(ValueError, match="speech endpoints"):
        Settings(**base, stt_base_url="http://stt.internal/v1")
    with pytest.raises(ValueError, match="embedded credentials"):
        Settings(demo_token=SecretStr(TOKEN), tts_base_url="https://user:pw@tts.example")


# --- gateway unit tests ----------------------------------------------------------------


class FakeSTT:
    def __init__(self, text: str = QUESTION) -> None:
        self.text = text
        self.calls: list[tuple[int, int]] = []

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        self.calls.append((len(pcm), sample_rate))
        return self.text

    async def close(self) -> None:
        return None


class FakeTTS:
    def __init__(self, chunks: int = 3, hang: bool = False) -> None:
        self.chunks = chunks
        self.hang = hang
        self.texts: list[str] = []
        self.cancelled = False

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        self.texts.append(text)
        try:
            for _ in range(self.chunks):
                yield AudioChunk(tone(24000, 100), 24000)
                if self.hang:
                    await asyncio.sleep(30)
        except (asyncio.CancelledError, GeneratorExit):
            self.cancelled = True
            raise

    async def close(self) -> None:
        return None


def test_speakable_text_and_transcript_filter() -> None:
    answer = Answer(
        answer="Revenue was **USD 120 million** in 2024. [C1]\n\nMargin was 15%. [C2]",
        citations=[],
        abstained=False,
        profile=Profile.VOICE,
        retrieval_ms=1,
        generation_ms=1,
        request_id="r",
        model="m",
    )
    assert to_speakable(answer, 400) == "Revenue was USD 120 million in 2024. Margin was 15%."
    assert to_speakable(answer, 50) == "Revenue was USD 120 million in 2024."
    abstained = answer.model_copy(update={"answer": ABSTENTION, "abstained": True})
    assert to_speakable(abstained, 400) == ABSTENTION_PROMPT
    assert is_meaningful(QUESTION)
    assert not any(is_meaningful(text) for text in ["", "  ", ".", "Thank you.", "uh", "a"])


async def make_session(
    document: Document,
    stt: FakeSTT,
    tts: FakeTTS,
    events: list[VoiceEvent],
    await_ack: bool = False,
) -> VoiceSession:
    settings = Settings(demo_token=SecretStr(TOKEN), vad_min_speech_ms=100, vad_barge_in_ms=100)
    store = MemoryRetriever()
    await store.upsert(chunk_document(document))
    from prometheus_client import CollectorRegistry

    async def sink(event: VoiceEvent) -> None:
        events.append(event)

    deps = VoiceDependencies(
        settings=settings,
        service=RagService(settings, store, make_provider(settings)),
        stt=stt,
        tts=tts,
        metrics=VoiceMetrics(CollectorRegistry()),
    )
    principal = Principal(subject="caller", tenant_id="demo", roles=["reader"])
    return VoiceSession(
        deps,
        principal,
        sink,
        input_rate=16000,
        output_rate=16000,
        await_playback_ack=await_ack,
        transport="test",
    )


async def eventually(predicate: Any) -> None:
    for _ in range(500):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached within 5 s")


async def test_session_turn_and_barge_in(document: Document) -> None:
    events: list[VoiceEvent] = []
    stt, tts = FakeSTT(), FakeTTS(hang=True)
    session = await make_session(document, stt, tts, events)
    await session.feed(tone(16000, 300) + silence(16000, 600))
    await eventually(lambda: any(isinstance(e, AudioEvent) for e in events))
    assert session.state is VoiceState.SPEAKING
    assert stt.calls[0][1] == 16000
    assert isinstance(events[0], TranscriptEvent) and events[0].text == QUESTION
    answer = next(e for e in events if isinstance(e, AnswerEvent))
    assert answer.answer is not None and not answer.answer.abstained
    assert "[C1]" not in answer.speakable and tts.texts == [answer.speakable]
    await session.feed(tone(16000, 200))
    assert isinstance(events[-1], ClearEvent)
    assert session.state is VoiceState.LISTENING and tts.cancelled
    assert session.turns[-1].outcome == "interrupted"
    assert session.turns[-1].first_audio_ms is not None
    await session.close()


async def test_session_ignores_garbage_and_speaks_abstention(document: Document) -> None:
    events: list[VoiceEvent] = []
    session = await make_session(document, FakeSTT("Thank you."), FakeTTS(), events)
    await session.feed(tone(16000, 300) + silence(16000, 600))
    await eventually(lambda: bool(session.turns))
    assert session.turns[-1].outcome == "ignored" and events == []

    events.clear()
    tts = FakeTTS()
    session = await make_session(document, FakeSTT("Who is the CEO of Mars?"), tts, events)
    await session.feed(tone(16000, 300) + silence(16000, 600))
    await eventually(lambda: any(isinstance(e, TurnMetricsEvent) for e in events))
    assert tts.texts == [ABSTENTION_PROMPT]
    metrics = next(e for e in events if isinstance(e, TurnMetricsEvent)).metrics
    assert metrics.outcome == "abstained" and metrics.abstained
    assert metrics.stt_ms is not None and metrics.retrieval_ms is not None
    assert metrics.first_audio_ms is not None and metrics.total_ms is not None
    assert any(isinstance(e, MarkEvent) for e in events)
    # 3 x 100 ms of 24 kHz audio resampled to 16 kHz
    audio = b"".join(e.pcm for e in events if isinstance(e, AudioEvent))
    assert abs(len(audio) // 2 - 4800) <= 3
    await session.close()


# --- websocket transports --------------------------------------------------------------


@pytest.fixture
def voice_client(document: Document) -> Iterator[tuple[TestClient, FakeSTT, FakeTTS]]:
    settings = Settings(demo_token=SecretStr(TOKEN), voice_enabled=True)
    stt, tts = FakeSTT(), FakeTTS()
    app = create_app(settings, MemoryRetriever(), make_provider(settings), stt=stt, tts=tts)
    with TestClient(app) as client:
        response = client.post(
            "/v1/documents", json={"documents": [document.model_dump(mode="json")]}, headers=AUTH
        )
        assert response.status_code == 201
        yield client, stt, tts


def start_event(token: str | None = TOKEN) -> dict[str, Any]:
    parameters = {"token": token} if token else {}
    return {
        "event": "start",
        "sequenceNumber": "1",
        "streamSid": "MZ123",
        "start": {
            "streamSid": "MZ123",
            "callSid": "CA123",
            "tracks": ["inbound"],
            "customParameters": parameters,
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
        },
    }


def twilio_media(pcm: bytes) -> Iterator[dict[str, Any]]:
    encoded = mulaw_encode(pcm)
    for index in range(0, len(encoded), 160):
        yield {
            "event": "media",
            "streamSid": "MZ123",
            "media": {
                "track": "inbound",
                "payload": base64.b64encode(encoded[index : index + 160]).decode(),
            },
        }


def send_utterance(ws: Any) -> None:
    for message in twilio_media(tone(8000, 400) + silence(8000, 700)):
        ws.send_json(message)


@pytest.mark.parametrize("token", [None, "wrong-token-value"])
def test_twilio_rejects_unauthenticated_streams(
    voice_client: tuple[TestClient, FakeSTT, FakeTTS], token: str | None
) -> None:
    client, stt, _ = voice_client
    with client.websocket_connect("/v1/voice/twilio") as ws:
        ws.send_json({"event": "connected", "protocol": "Call", "version": "1.0.0"})
        ws.send_json(start_event(token))
        with pytest.raises(WebSocketDisconnect) as error:
            ws.receive_json()
    assert error.value.code == 1008
    with client.websocket_connect("/v1/voice/twilio") as ws:
        next_message = next(twilio_media(tone(8000, 20)))
        ws.send_json(next_message)  # media before start/auth
        with pytest.raises(WebSocketDisconnect) as error:
            ws.receive_json()
    assert error.value.code == 1008
    with pytest.raises(WebSocketDisconnect) as error:
        with client.websocket_connect(
            "/v1/voice/twilio", headers={"origin": "https://evil.example"}
        ) as ws:
            ws.receive_json()
    assert error.value.code == 1008
    assert stt.calls == []


def test_twilio_turn_media_mark_and_metrics(
    voice_client: tuple[TestClient, FakeSTT, FakeTTS],
) -> None:
    client, stt, tts = voice_client
    with client.websocket_connect("/v1/voice/twilio") as ws:
        ws.send_json({"event": "connected"})
        ws.send_json(start_event())
        send_utterance(ws)
        frames: list[bytes] = []
        while True:
            message = ws.receive_json()
            assert message["streamSid"] == "MZ123"
            if message["event"] == "mark":
                break
            assert message["event"] == "media"
            frames.append(base64.b64decode(message["media"]["payload"]))
        assert {len(frame) for frame in frames} == {160}
        # 300 ms of agent audio at 8 kHz -> 15 x 20 ms frames
        assert len(frames) == 15
        ws.send_json({"event": "mark", "streamSid": "MZ123", "mark": message["mark"]})
        ws.send_json({"event": "stop", "streamSid": "MZ123"})
        with pytest.raises(WebSocketDisconnect) as error:
            ws.receive_json()
        assert error.value.code == 1000
    assert stt.calls[0][1] == 8000
    assert tts.texts == ["Revenue was USD 120 million in 2024."]
    metrics = client.get("/metrics", headers=AUTH).text
    assert "ledger_voice_first_audio_seconds_count 1.0" in metrics
    assert "ledger_voice_stt_seconds_count 1.0" in metrics
    assert 'ledger_voice_turns_total{outcome="answered"} 1.0' in metrics
    assert 'ledger_retrieval_seconds_count{profile="voice"} 1.0' in metrics
    assert 'ledger_voice_sessions_total{outcome="completed",transport="twilio"} 1.0' in metrics


def test_twilio_barge_in_emits_clear(voice_client: tuple[TestClient, FakeSTT, FakeTTS]) -> None:
    client, _, tts = voice_client
    tts.hang = True
    with client.websocket_connect("/v1/voice/twilio") as ws:
        ws.send_json({"event": "connected"})
        ws.send_json(start_event())
        send_utterance(ws)
        assert ws.receive_json()["event"] == "media"
        for message in twilio_media(tone(8000, 400)):
            ws.send_json(message)
        while (message := ws.receive_json())["event"] == "media":
            pass
        assert message == {"event": "clear", "streamSid": "MZ123"}
    assert tts.cancelled
    metrics = client.get("/metrics", headers=AUTH).text
    assert "ledger_voice_barge_ins_total 1.0" in metrics
    assert 'ledger_voice_turns_total{outcome="interrupted"} 1.0' in metrics


def test_voice_stream_endpoint(voice_client: tuple[TestClient, FakeSTT, FakeTTS]) -> None:
    client, stt, _ = voice_client
    with pytest.raises(WebSocketDisconnect) as error:
        with client.websocket_connect("/v1/voice/stream") as ws:
            ws.receive_json()
    assert error.value.code == 1008
    with client.websocket_connect("/v1/voice/stream", headers=AUTH) as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready" and ready["sample_rate"] == 16000
        speech = tone(16000, 400)
        for index in range(0, len(speech), 640):
            ws.send_bytes(speech[index : index + 640])
        ws.send_text(json.dumps({"type": "end"}))  # push-to-talk endpoint, no trailing silence
        transcript = ws.receive_json()
        assert transcript["type"] == "transcript" and transcript["text"] == QUESTION
        answer = ws.receive_json()
        assert answer["type"] == "answer" and not answer["abstained"]
        assert answer["text"] == "Revenue was USD 120 million in 2024."
        assert answer["citations"][0]["document_id"] == "doc-1"
        audio = b""
        while "bytes" in (message := ws.receive()) and message["bytes"] is not None:
            audio += message["bytes"]
        assert json.loads(message["text"]) == {"type": "audio_end", "turn_id": answer["turn_id"]}
        assert abs(len(audio) // 2 - 4800) <= 3
        metrics = ws.receive_json()
        assert metrics["type"] == "turn_metrics" and metrics["outcome"] == "answered"
        for key in [
            "stt_ms",
            "retrieval_ms",
            "generation_ms",
            "tts_first_byte_ms",
            "first_audio_ms",
        ]:
            assert metrics[key] is not None and metrics[key] >= 0
        ws.send_text(json.dumps({"type": "bogus"}))
        assert ws.receive_json()["type"] == "error"
        ws.send_bytes(b"\x00" * 40000)
        with pytest.raises(WebSocketDisconnect) as error:
            ws.receive_json()
        assert error.value.code == 1009
    assert stt.calls[0][1] == 16000


def test_voice_disabled_closes(settings: Settings) -> None:
    app = create_app(settings, MemoryRetriever(), make_provider(settings))
    with TestClient(app) as client:
        for path in ["/v1/voice/twilio", "/v1/voice/stream"]:
            with pytest.raises(WebSocketDisconnect) as error:
                with client.websocket_connect(path, headers=AUTH) as ws:
                    ws.receive_json()
            assert error.value.code == 1008
