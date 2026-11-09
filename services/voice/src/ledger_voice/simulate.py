"""Twilio Media Streams simulator: a phone call without a phone (ADR 0006 gate 5).

Streams a mono WAV file to ``/twilio/media`` exactly as Twilio would (8 kHz
mu-law, 20 ms frames, real time), keeps sending silence while waiting, and
measures **end of caller audio -> first audio byte back**. The reply is saved as
a WAV file so it can be listened to. Needs the same ``LEDGER_VOICE_STREAM_SECRET``
as the service (run both through ``ledgerctl run``).

    say -o question.wav --data-format=LEI16@8000 "What was Aurora revenue in 2024?"
    ledgerctl run -- uv run --package ledger-voice ledger-voice-sim question.wav
"""

import argparse
import asyncio
import base64
import json
import os
import secrets
import sys
import time
import wave
from pathlib import Path

import numpy as np
import soxr
import websockets

from ledger_voice.tokens import mint

RATE = 8000
FRAME_SAMPLES = 160  # 20 ms at 8 kHz

# G.711 mu-law (ITU-T) encode/decode, vectorised; audioop is gone from Python 3.13.
_BIAS, _CLIP = 0x84, 32635


def mulaw_encode(pcm: np.ndarray) -> bytes:
    samples = np.clip(pcm.astype(np.int32), -_CLIP, _CLIP)
    sign = np.where(samples < 0, 0x80, 0)
    magnitude = np.abs(samples) + _BIAS
    exponent = np.clip(np.floor(np.log2(magnitude)).astype(np.int32) - 7, 0, 7)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    encoded: bytes = (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8).tobytes()
    return encoded


def mulaw_decode(data: bytes) -> np.ndarray:
    value = ~np.frombuffer(data, dtype=np.uint8).astype(np.int32) & 0xFF
    sign, exponent, mantissa = value & 0x80, (value >> 4) & 0x07, value & 0x0F
    magnitude = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
    return np.where(sign != 0, -magnitude, magnitude).astype(np.int16)


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as stream:
        if stream.getsampwidth() != 2:
            raise SystemExit("WAV must be 16-bit PCM")
        frames = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16)
        channels, rate = stream.getnchannels(), stream.getframerate()
    if channels > 1:
        frames = frames.reshape(-1, channels).mean(axis=1).astype(np.int16)
    if rate != RATE:
        frames = soxr.resample(frames, rate, RATE).astype(np.int16)
    return frames


async def simulate(
    url: str, audio: np.ndarray, secret: str, wait: float, output: Path
) -> dict[str, float | int | None]:
    call_sid = "CA" + secrets.token_hex(16)
    stream_sid = "MZ" + secrets.token_hex(16)
    token = mint(secret, call_sid, 60)
    received = bytearray()
    first_audio_at: float | None = None
    cleared = 0
    closed_code: int | None = None
    async with websockets.connect(url, max_size=1_048_576) as ws:
        await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        await ws.send(
            json.dumps(
                {
                    "event": "start",
                    "streamSid": stream_sid,
                    "start": {
                        "streamSid": stream_sid,
                        "callSid": call_sid,
                        "accountSid": "ACsimulator",
                        "customParameters": {"token": token},
                        "mediaFormat": {
                            "encoding": "audio/x-mulaw",
                            "sampleRate": RATE,
                            "channels": 1,
                        },
                    },
                }
            )
        )

        async def receive() -> None:
            nonlocal first_audio_at, cleared
            async for raw in ws:
                message = json.loads(raw)
                if message.get("event") == "media":
                    if first_audio_at is None:
                        first_audio_at = time.perf_counter()
                    received.extend(base64.b64decode(message["media"]["payload"]))
                elif message.get("event") == "clear":
                    cleared += 1

        listener = asyncio.create_task(receive())
        silence = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        frames = [audio[i : i + FRAME_SAMPLES] for i in range(0, len(audio), FRAME_SAMPLES)]
        start = time.perf_counter()
        sequence = 0
        speech_end = 0.0
        total = len(frames) + int(wait * 1000 / 20)
        for index in range(total):
            if ws.close_code is not None:
                closed_code = ws.close_code
                break
            chunk = frames[index] if index < len(frames) else silence
            if len(chunk) < FRAME_SAMPLES:
                chunk = np.pad(chunk, (0, FRAME_SAMPLES - len(chunk)))
            sequence += 1
            media = {
                "event": "media",
                "sequenceNumber": str(sequence),
                "streamSid": stream_sid,
                "media": {
                    "track": "inbound",
                    "payload": base64.b64encode(mulaw_encode(chunk)).decode(),
                },
            }
            try:
                await ws.send(json.dumps(media))
            except websockets.ConnectionClosed:
                closed_code = ws.close_code
                break
            if index == len(frames) - 1:
                speech_end = time.perf_counter()
            # Real-time pacing against the wall clock, as Twilio does.
            await asyncio.sleep(max(0.0, start + (index + 1) * 0.02 - time.perf_counter()))
        if closed_code is None:
            try:
                await ws.send(json.dumps({"event": "stop", "streamSid": stream_sid}))
            except websockets.ConnectionClosed:
                closed_code = ws.close_code
        listener.cancel()
    with wave.open(str(output), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(mulaw_decode(bytes(received)).tobytes())
    latency = None if first_audio_at is None else round((first_audio_at - speech_end) * 1000, 1)
    return {
        "end_of_audio_to_first_reply_ms": latency,
        "reply_seconds": round(len(received) / RATE, 2),
        "clear_events": cleared,
        "server_closed_early_with": closed_code,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="ledger-voice-sim", description=__doc__.split("\n\n")[0])
    parser.add_argument("audio", type=Path, help="mono 16-bit WAV with the caller's question")
    parser.add_argument("--url", default="ws://127.0.0.1:8090/twilio/media")
    parser.add_argument("--wait", type=float, default=12.0, help="seconds of silence to wait")
    parser.add_argument("--output", type=Path, default=Path("reply.wav"))
    args = parser.parse_args()
    secret = os.environ.get("LEDGER_VOICE_STREAM_SECRET")
    if not secret:
        raise SystemExit("LEDGER_VOICE_STREAM_SECRET is not set; run through `ledgerctl run`")
    result = asyncio.run(simulate(args.url, load_wav(args.audio), secret, args.wait, args.output))
    json.dump(result, sys.stdout, indent=2)
    print(f"\nreply audio: {args.output}")
