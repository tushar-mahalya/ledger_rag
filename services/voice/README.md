# ledger-voice

Voice channel for Ledger ([ADR 0006](../../docs/adr/0006-voice-channel-service.md)).
It takes calls over Twilio Media Streams, builds a Pipecat pipeline with **no LLM**,
asks the engine's public API (`POST /v1/query`, `profile=voice`) and speaks only the
validated answer or a fixed phrase.

```
Twilio --> /twilio/voice (signature check, per-call token)
       --> /twilio/media --> VAD --> Scribe v2 Realtime --> EngineAnswer --> Flash v2.5 --> caller
                                                               |
                                                engine /v1/query (profile=voice)
```

| Module | Role |
|---|---|
| `app.py` | FastAPI service: Twilio webhook, media WebSocket, health, session limits |
| `answer.py` | Turn detection plus the only producer of speech (drops every other text frame) |
| `engine.py` | Engine contract client, speakable-text rules, fixed phrases |
| `pipeline.py` | Pipeline builder and the guard that rejects any LLM or misordered speech path |
| `tokens.py` | Short-lived, single-use, call-bound stream tokens |
| `logs.py` | Logging floor at INFO (Pipecat logs transcripts at DEBUG) |
| `simulate.py` | Twilio simulator: plays a WAV as a call, measures end of speech to first reply audio |
| `modal_app.py` | Modal deployment (not deployed yet) |

## Run locally

Settings come from `config/env/dev.env` and the SOPS store via `ledgerctl`
(`LEDGER_VOICE_*`). Start the engine, then this service, then simulate a call:

```bash
ledgerctl run -- uv run uvicorn ledger_rag.api.app:app --port 8080 --no-access-log
ledgerctl run -- uv run --package ledger-voice ledger-voice          # 127.0.0.1:8090
say -o question.wav --data-format=LEI16@8000 "What was Aurora revenue in 2024?"
ledgerctl run -- uv run --package ledger-voice ledger-voice-sim question.wav
```

Live calls need `LEDGER_ELEVENLABS_API_KEY` (and a voice id). Real phone calls also
need `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, a public HTTPS URL
(`LEDGER_VOICE_PUBLIC_URL`, e.g. a `cloudflared` tunnel) and the dialled number in
`LEDGER_VOICE_ALLOWED_NUMBERS`. Point the Twilio number's voice webhook at
`<public URL>/twilio/voice` (HTTP POST).

## Tests

```bash
uv run pytest services/voice/tests
```

The ADR's live gates (15-minute Modal sessions, measured latency, real barge-in on a
call) are still open.
