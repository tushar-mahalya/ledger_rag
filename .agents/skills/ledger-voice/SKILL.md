---
name: ledger-voice
description: "Use when changing or debugging Ledger's phone channel in services/voice: Twilio webhook and Media Streams handling, the pinned Pipecat 1.14 pipeline, ElevenLabs STT/TTS settings, barge-in or the Modal voice app. Not for engine retrieval, chat or MCP work."
---

# Ledger voice channel

Design: [ADR 0006](../../../docs/adr/0006-voice-channel-service.md),
[service README](../../../services/voice/README.md). Code:
[webhook and media socket](../../../services/voice/src/ledger_voice/app.py),
[pipeline guard](../../../services/voice/src/ledger_voice/pipeline.py),
[answer step](../../../services/voice/src/ledger_voice/answer.py),
[log floor](../../../services/voice/src/ledger_voice/logs.py),
[Modal app](../../../services/voice/modal_app.py).

## Use the pinned APIs, not memory

- `pipecat-ai` is pinned to 1.14.0, while docs.pipecat.ai tracks the newest
  release (1.15.0 on 2026-11-09). Read the installed `pipecat` package for exact
  signatures; use the [docs index](https://docs.pipecat.ai/llms.txt) for concepts.
- `PipelineWorker` (`pipecat.pipeline.worker`) and `WorkerRunner`
  (`pipecat.workers.runner`) replace `PipelineTask` and `PipelineRunner`
  (deprecated since 1.3.0, removed in 2.0.0). Call `add_workers()` before `run()`.
- ElevenLabs services take `<Service>.Settings(...)`; their `InputParams` is
  deprecated. `TwilioFrameSerializer` still takes `InputParams`.
- Twilio: query the `twilio-docs` MCP server from `.mcp.json`, or read
  [Media Streams messages](https://www.twilio.com/docs/voice/media-streams/websocket-messages)
  and [`<Stream>`](https://www.twilio.com/docs/voice/twiml/stream). ElevenLabs has
  no docs MCP server: see [commit strategies](https://elevenlabs.io/docs/eleven-api/guides/how-to/speech-to-text/realtime/transcripts-and-commit-strategies)
  and [models](https://elevenlabs.io/docs/overview/models).

## Rules the code depends on

- Speech path: transport in, `VADProcessor`, STT, `EngineAnswerProcessor`, the
  only TTS, transport out. `assert_speech_path` rejects any `LLMService` or other
  order. Do not add an LLM, a second TTS or a processor that forwards text.
- `TranscriptionFrame` subclasses `TextFrame`; `TTSSpeakFrame` does not. The
  answer step drops every downstream `TextFrame` and speaks one `TTSSpeakFrame`
  holding the validated answer or a fixed phrase.
- Each processor handles `SystemFrame`s (`InterruptionFrame`, `CancelFrame`, VAD
  and turn events) before queued data and control frames (transcripts,
  `TTSSpeakFrame`, `EndFrame`). Turn ends are decided in the `UserTurnProcessor`
  subclass that collects transcripts; deciding them elsewhere races the last one.
- `ElevenLabsRealtimeSTTService` defaults to manual commits triggered by
  `VADUserStoppedSpeakingFrame` from the upstream `VADProcessor`. Removing VAD or
  switching commit strategy changes turn timing; re-measure latency.
- `<Connect><Stream>` is bidirectional. Its `wss` URL cannot carry a query
  string, so the per-call token travels as a `<Parameter>` and arrives in
  `start.customParameters`. `_await_start` verifies it before any `media`;
  WebSockets with an `Origin` header are refused.
- Validate `X-Twilio-Signature` with the SDK `RequestValidator` against
  `LEDGER_VOICE_PUBLIC_URL` plus the path. Twilio signs the webhook URL set in
  Twilio, which `request.url` may not reproduce behind `cloudflared` or Modal.
- Media is base64 `audio/x-mulaw` at 8000 Hz. `TwilioFrameSerializer` converts
  audio and sends `clear` on `InterruptionFrame`; the answer step cancels its
  engine task on interruption, end or cancel. `auto_hang_up` calls Twilio REST
  only for the configured account.
- Pipecat logs transcripts at DEBUG. Keep the INFO floor; log counts and timings,
  never transcripts, answers, audio or tokens.
- Twilio media connects straight to this service on Modal, not through
  Cloudflare Workers.

## Checks

```bash
uv run --locked pytest -q services/voice/tests
```

`ledger-voice-sim` streams a WAV through the local service with live ElevenLabs
(billable) and the engine. It is not phone-network latency. Barge-in on a real
call, 15-minute Modal sessions and measured latency are open ADR gates; mocks
and simulator runs do not close them.
