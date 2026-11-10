# ADR 0009: Stream lifecycle events and one validated answer, not model tokens

- Status: accepted, as recorded in [project instructions](../../AGENTS.md). No separate approval record exists
- Date: unknown. Commit `ff349fc` (2026-10-31) is the earliest recorded statement found
- Evidence basis: recorded intent for the choice and its main reason. The option comparison is a present-day analysis

## Context

Chat and voice share one engine. The engine validates citations and numbers only on a complete draft.
Voice must not speak unvalidated model output ([AGENTS.md](../../AGENTS.md)).
Clients still need progress signals while retrieval, tools and validation run.

## Decision

`POST /v1/query/stream` sends `activity` and `evidence` events, then one validated `answer`, then `done`.
It never forwards raw model tokens or private reasoning.
A failure sends `error` and no `answer` or `done`.

## Rationale

The README records the reason: streaming the complete validated answer avoids speaking an answer before the citation gate ([README](../../README.md#what-actually-runs)).
The same rule keeps chat and voice on one validation path.

## Options

| Option | Benefits | Costs and risks | Evidence | Outcome |
|---|---|---|---|---|
| Activity and evidence events, then one validated answer | Nothing unvalidated reaches a client or speaker | The client waits for the full draft and validation | [app.py](../../services/engine/src/ledger_rag/api/app.py#L313-L362), [CHAT_API.md](../CHAT_API.md#sse-wire-contract) | Selected |
| Raw model token streaming | Lowest time to first visible text | Clients would show or speak text that may later abstain | [AGENTS.md](../../AGENTS.md), [README](../../README.md#what-actually-runs) | Rejected by recorded intent |
| Validated-sentence streaming as an additive event | Voice could start speech on the first validated sentence | Needs per-sentence validation as a stable contract | [ADR 0006](0006-voice-channel-service.md) | Proposed follow-up; not evaluated |

## Consequences

- Clients accept an answer only after `done` ([api.ts](../../services/web/src/api.ts#L100-L160)).
- Time to first answer text includes generation and validation. No latency measurement for this path was found.
- An abstention never shows a rejected draft ([service.py](../../services/engine/src/ledger_rag/orchestration/service.py#L237-L242)).

## Reversal cost

Token streaming would need a new event type and client changes in the web workspace and voice service.
It would also need a rule for retracting text that later fails validation.
The contract fixtures in [contracts](../../contracts/README.md) would change.

## Validation

The [stream contract tests](../../services/engine/tests/integration/test_chat_stream.py#L36-L115) pass locally with fake providers.
User-perceived latency is unmeasured.

## Revisit when

- Measured time to first answer text exceeds a stated chat or voice target.
- The engine exposes a stable per-sentence validation result.

## Links

- Implementation: [stream route](../../services/engine/src/ledger_rag/api/app.py#L313-L362).
- Journey: [chat query](../architecture/runtime/chat-query-stream.md).
