---
name: ledger-voice
description: "Use when changing IVR/WebSocket audio, VAD, barge-in, speech endpoints, turn cancellation, TTS streaming or voice latency in Ledger RAG."
---

# Low-latency voice without stale or ungrounded speech

## Read

[Voice package](../../../src/ledger_rag/voice),
[voice tests](../../../tests/unit/test_voice.py),
[service](../../../src/ledger_rag/orchestration/service.py).

## Procedure

1. Trace one turn: authenticated media -> codec/VAD -> STT -> voice-profile RAG
   -> citation gate -> TTS -> transport. Separate implemented gateway behavior
   from unverified provider/telephony interoperability.
2. Preserve the voice profile's bounded evidence and exclusion of chat tools
   and reranking. Do not add network hops to the hot path without measured benefit.
3. Test 20-ms frames, codec/sample-rate validation, silence, short speech,
   maximum utterance/session duration, malformed frames and disconnects.
4. Barge-in must cancel stale work, prevent stale audio playback and release
   tasks/permits. Test interruption during STT, RAG and TTS, not only idle silence.
5. Bound transcript length, TTS duration/buffers and provider timeouts; sanitize
   failures. Never log transcripts or raw audio by default.
6. Streaming TTS does not mean streaming unvalidated LLM tokens. Keep the evidence
   gate before speech. Distinguish application speech latency from telephone
   playback latency and endpointing delay.
7. Measure end-of-user-speech to first playable audio, warm/cold p95, cancellations,
   interruptions and failures with real endpoints before claiming a latency SLO.
   Local mocks only establish lifecycle correctness.

```bash
uv run --locked pytest -q tests/unit/test_voice.py tests/integration/test_resilience.py
```

## Done / stop

Return lifecycle regressions covered and a stage-level latency budget/results.
Stop before live calls without credentials, consent, cost approval and the
provider's current authentication protocol. Semantic turn detection is a
research track, not a feature already wired into this gateway.
