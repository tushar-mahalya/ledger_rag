# ADR 0006: Voice becomes a separate channel service

- Status: accepted; implementation in progress in [`services/voice`](../../services/voice/).
  The in-process gateway is removed only after the gates below pass
- Date: 2026-11-09

## Context

Today voice is ~1,500 lines in `src/ledger_rag/voice/`, mounted as a router in the
engine's FastAPI process. It calls `RagService` in-process and shares the engine's
settings, rate limiter, metrics and process lifetime. It uses energy VAD, then
uploads each finished utterance to an OpenAI-compatible batch STT endpoint, and
streams OpenAI-compatible TTS. Twilio sessions authenticate with a
custom-parameter token whose signed minting flow was never built. Only synthetic
lifecycle tests exist.

The product goal is one evidence-first engine serving several channels (IVR,
chat workspace, MCP, future integrations) with the lowest end-to-end latency the
invariants allow, vendor bake-offs for speech, and a single maintainer.

## Decision

1. **Process and deployment boundary.** Voice runs as its own service,
   `ledger-voice`: its own container or Modal app, dependencies, scaling, deploys
   and secrets route. It stays in this repository (a separate workspace package
   with the same lockfile, CI and `ledgerctl`), so contract changes are atomic.
2. **Contract.** Voice is an ordinary client of the engine's public HTTP API:
   `POST /v1/query` with `profile=voice`. The engine keeps enforcing the voice
   bounds server-side (at most three chunks, no tools or reranker, deadlines,
   validated output only), whichever client calls. Voice keeps its own small
   typed client models, checked against generated engine fixtures in CI the
   same way the web workspace is. It imports no engine internals.
3. **Identity.** Voice authenticates to the engine as a per-line service
   principal: each phone line maps to one tenant's client credential with the
   `reader` role. Caller-supplied tenant or group data is never forwarded. Twilio
   connections are verified (`X-Twilio-Signature` plus a per-call token) before
   any media is processed.
4. **Media pipeline.** Build on [Pipecat](https://pypi.org/project/pipecat-ai/)
   (BSD-2-Clause, pinned to an exact version) for the WebSocket transport, Twilio
   serializer, STT/TTS services, turn detection and interruptions. Ledger owns:
   - the answer step: a frame processor turns a final transcription into one
     engine call and speaks only the validated answer text or a fixed abstention.
     The pipeline contains **no LLM service**, so no model token can reach TTS;
   - session and frame bounds, connection authentication and redacted telemetry.

   Start with ElevenLabs Scribe v2 Realtime and Flash v2.5 TTS at 8 kHz PCM (Pipecat's
   Twilio serializer converts mu-law in and out), with `keyterms` from the company
   gazetteer. Pipecat 1.14's ElevenLabs service supports only `manual` (Pipecat VAD)
   and `vad` commits, so turns end by Silero VAD plus the Smart Turn v3 model owned by
   the answer step; ElevenLabs' `turn_prediction` mode is not exposed there. Other
   vendors become configuration for the bake-off.
5. **Placement.** Run on Modal CPU containers in the same region as the engine,
   Qdrant and the Twilio processing region, warm during test windows. Twilio
   media connects directly to this service, not through Cloudflare Workers.
   Cloudflare's role (web workspace, public HTTP front door) is a separate decision.
6. **Migration.** The in-process gateway stays until the new service passes every
   gate. It is then removed from the engine together with the `voice_*` settings.
   Production never runs both paths.

## Trade-offs

| Concern | In-process (today) | Separate service (chosen) |
|---|---|---|
| Latency | No hop between voice and engine | One extra same-region HTTPS call on a kept-alive connection, expected in single-digit milliseconds against hundreds per STT, LLM and TTS stage. An estimate until gate 5 measures it |
| Failure isolation | An engine deploy drops live calls; a call surge consumes query admission | Independent deploys and draining; separate admission limits |
| Least privilege | The internet-facing media process holds Gemini, Qdrant and FRED keys | Holds only the ElevenLabs key, Twilio auth token and its engine client credential |
| Dependencies | Audio/ML packages would enter the engine image | Pipecat's tree (numpy, numba, onnxruntime, ...) stays in the voice image |
| Scaling | One unit for 15-minute sessions and sub-second queries | Sessions scale by concurrent calls; the engine by query load |
| Reuse | Voice uses private in-process calls | Voice proves the public contract works for a real-time channel |
| Cost of the split | One deployable | Two deployables, service auth, contract versioning, cross-service tracing |

## Alternatives considered

| Option | Verdict |
|---|---|
| Keep voice in-process | Acceptable for a single-container demo; rejected for the multi-channel product for the reasons above |
| Separate repository | Rejected: version skew and duplicated CI/secrets tooling for one maintainer; a monorepo keeps contract changes atomic |
| Evolve the custom gateway | Viable fallback if a Pipecat gate fails. Not primary: each vendor needs a bespoke streaming client, turn detection and barge-in must be hand-built, and the bake-off costs more |
| [LiveKit Agents](https://www.evalgent.com/blog/pipecat-vs-livekit) | WebRTC-first and reaches the phone network through SIP plus a LiveKit server. More infrastructure than Twilio Media Streams needs; revisit for browser-first voice |
| Managed agents (ElevenLabs Agents, Vapi, Retell) with a custom-LLM hook | Fastest demo, but the vendor owns turn-taking and telemetry, bills per minute and processes the data, and stage latencies are hard to measure. Kept only as a possible measured baseline |
| Speech-to-speech models (Gemini Live, realtime APIs) | Rejected for answers: they speak model output before citation validation. Usable as an STT-only arm |
| Gateway on Cloudflare Workers/Containers | Rejected for the media path now. Twilio streams from its processing region, not from the caller, so edge placement adds nothing, while it adds a cross-cloud hop to the engine and GPUs |

Pipecat risks: weekly releases (1.8.1 on 2026-08-27 to 1.15.0 on 2026-10-29),
including changed interruption semantics in 1.11. It is therefore pinned exactly
and upgraded only after the voice regression suite passes.

## Gates before removing the in-process gateway

1. **Validated speech only.** A test asserts the pipeline has no LLM service and
   that TTS receives only engine-validated text or the fixed abstention.
2. **Security parity.** Twilio signature and per-call token are checked before
   media; browser-origin WebSockets are rejected; the current frame, session,
   transcript and deadline bounds are ported with their tests.
3. **Barge-in.** Caller speech during playback sends Twilio `clear` and cancels
   in-flight engine and TTS work, releasing session capacity.
4. **Long sessions on Modal.** A 15-minute WebSocket session survives with the
   function timeout above the session limit. Modal's documentation pages checked
   on 2026-11-09 state no WebSocket duration or idle limit, so this is measured.
5. **Latency.** End-of-speech to first audio byte sent to Twilio is measured per
   stage (turn commit, engine, TTS first byte, transport) on recorded 8 kHz
   financial questions, reported as p50/p95 with sample counts. The engine hop
   is included.
6. **Contract.** CI checks the engine `Answer` fixtures against the voice
   client models.

## Consequences and follow-ups

- `services/voice` (package `ledger_voice`) with a Modal app and the `modal-voice`
  secret target carrying only the speech key, Twilio credentials, the stream-token
  key and its engine credential. Implemented 2026-11-09: signed webhook with per-call
  single-use tokens, Pipecat pipeline without an LLM and a structural guard, answer
  step that owns turn detection (avoiding Pipecat's system-before-data frame race),
  INFO logging floor (Pipecat logs transcripts at DEBUG) and a Twilio simulator for
  gate 5. Not yet exercised with live ElevenLabs, Twilio or Modal.
- [Architecture](../ARCHITECTURE.md#4-voice-lifecycle) marks the in-process
  gateway as transitional.
- Latency follow-up that keeps the invariant: the engine validates numbers per
  sentence already, so it could stream *validated sentences* (an additive SSE
  event). Voice would then start TTS on the first validated sentence instead of
  waiting for the whole answer.

Sources: [Pipecat on PyPI](https://pypi.org/project/pipecat-ai/),
[Pipecat supported services](https://docs.pipecat.ai/server/services/supported-services),
[Pipecat ElevenLabs STT](https://docs.pipecat.ai/api-reference/server/services/stt/elevenlabs),
[Pipecat custom frame processors](https://docs.pipecat.ai/pipecat/fundamentals/custom-frame-processor),
[ElevenLabs realtime STT](https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime),
[Twilio Media Streams security](https://www.twilio.com/docs/global-infrastructure/firewall-configurations/media-streams-configuration),
[Twilio edge locations](https://www.twilio.com/docs/global-infrastructure/understanding-edge-locations),
[Modal web endpoints](https://modal.com/docs/guide/webhooks),
[Modal timeouts](https://modal.com/docs/guide/timeouts).
