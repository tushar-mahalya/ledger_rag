# Understand Ledger RAG

This handbook explains Ledger RAG's boundaries, units and one priority journey with linked evidence.
It extends the existing documents in [docs](../). It does not replace them.

## Purpose, users and non-goals

- **Purpose:** answer financial-disclosure questions from authorized, cited evidence for chat, voice and MCP clients ([README](../../README.md)).
- **Users:** analysts in the web workspace, phone callers, MCP clients and operators ([context](context.md)).
- **Non-goals:** certified financial advice, a proven SOTA result and raw token streaming ([AGENTS.md](../../AGENTS.md)).

## Read first

1. [System context](context.md): actors and external systems.
2. [Deployable units](containers.md): the engine, clients, Modal workloads and `ledgerctl`.

## Trace a request

- [What happens when an analyst asks a question in the chat workspace?](runtime/chat-query-stream.md)

## Understand data

No handbook data page exists yet. [Ingestion](../INGESTION.md) is mapped as the data page DATA-001 but is not verified.
Also read [index contracts](../ARCHITECTURE.md#3-retrieval-and-index-contracts) and [ADR 0007](../adr/0007-corpus-pipeline.md).

## Understand choices

- Numbered records: [docs/adr](../adr/), mapped in place as ADR-0001 to ADR-0008. Record new decisions there.
- [DEC-001 - Stream lifecycle events and one validated answer](decisions/DEC-001-validated-answer-streaming.md) (handbook draft).
  It sits outside the project's ADR folder and numbering. A maintainer can convert it to an ADR.
- Evidence and alternatives: [research decisions](../research/2026-10-31-decisions.md).

## Operate the system

[Operations](../OPERATIONS.md), [secrets](../SECRETS.md), [validation record](../VALIDATION.md) and [agent workflows](../AGENT_WORKFLOWS.md).

## Coverage

| Scenario or concern | Priority | Page | Evidence | Gaps | Owner |
|---|---|---|---|---|---|
| Chat question over SSE | High | [chat-query-stream.md](runtime/chat-query-stream.md) | [app.py](../../services/engine/src/ledger_rag/api/app.py), [service.py](../../services/engine/src/ledger_rag/orchestration/service.py); integration tests ran | No real IdP, Qdrant, reranker or LLM exercised; latency unmeasured | unknown |
| System boundary and units | High | [context.md](context.md), [containers.md](containers.md) | READMEs, [compose.yaml](../../compose.yaml), entry points | Production topology and owners unknown | unknown |
| Validated-answer streaming | Medium | [DEC-001](decisions/DEC-001-validated-answer-streaming.md) | [README](../../README.md#what-actually-runs), [AGENTS.md](../../AGENTS.md) | Deciders and date unknown | unknown |
| Phone call through `ledger-voice` | High | Partly in [ARCHITECTURE.md](../ARCHITECTURE.md#4-voice-lifecycle) (not verified) | [ADR 0006](../adr/0006-voice-channel-service.md), [voice lifecycle](../ARCHITECTURE.md#4-voice-lifecycle) | No handbook journey; live calls unverified | unknown |
| Bulk index build and publication | High | [INGESTION.md](../INGESTION.md) (not verified) | [INGESTION.md](../INGESTION.md), [ADR 0007](../adr/0007-corpus-pipeline.md) | No handbook data lifecycle page | unknown |
| Document write, ACL change and erasure | Medium | Not covered | [app.py](../../services/engine/src/ledger_rag/api/app.py#L230-L292) | Lease, reconciliation and deletion proof not traced | unknown |
| MCP tools | Medium | [MCP.md](../MCP.md) (not verified) | [MCP.md](../MCP.md), [ADR 0004](../adr/0004-mcp-adapter.md) | Not traced | unknown |
| Secrets delivery | Medium | [SECRETS.md](../SECRETS.md) (not verified) | [SECRETS.md](../SECRETS.md), [ADR 0005](../adr/0005-sops-age-secrets.md) | Not traced | unknown |
| Observability and recovery | Medium | Partly in [chat-query-stream.md](runtime/chat-query-stream.md#observability-and-recovery) | [OPERATIONS.md](../OPERATIONS.md) | No alert rules or dashboards exist | unknown |
| Evaluation and training | Low | Not covered | [EVALUATION.md](../EVALUATION.md) | Outside runtime architecture | unknown |

This coverage is a sample. It does not claim complete execution coverage.

## Existing documents

`map.json` maps these documents in place with narrow `covers`, so Archlens tracks their freshness.
Each one stays where it is and has the review state `needs-review`.

| Document | Map ID | Covers | Verification on 2026-11-10 |
|---|---|---|---|
| [ARCHITECTURE.md](../ARCHITECTURE.md) | ARCH-001 | Boundaries, query planning, index contracts, voice lifecycle, release gates | Sections 2 and "Query planning" checked against source; no contradiction found |
| [CHAT_API.md](../CHAT_API.md) | API-001 | Request, SSE events, grounded calculation, graph budgets | Request, SSE, abstention and graph sections checked; one statement fixed (see below) |
| [CHAT_WORKSPACE.md](../CHAT_WORKSPACE.md) | WEB-001 | Web workspace architecture and trust contracts | Not verified |
| [INGESTION.md](../INGESTION.md) | DATA-001 | Modal ingestion, chunk profiles, index release | Not verified |
| [OPERATIONS.md](../OPERATIONS.md) | OPS-001 | Production status, telemetry, capacity, backup | Not verified |
| [SECRETS.md](../SECRETS.md) | SEC-001 | SOPS + age store and `ledgerctl` delivery | Not verified |
| [MCP.md](../MCP.md) | MCP-001 | MCP adapter | Not verified |
| [EXTERNAL_DATA.md](../EXTERNAL_DATA.md) | EXT-001 | SEC and FRED evidence | Not verified |
| [ADR 0001](../adr/0001-native-bm25.md) to [ADR 0008](../adr/0008-program-of-thought-reasoning.md) | ADR-0001 to ADR-0008 | Numbered decision records | Status lines read; content not verified |

One `CHAT_API.md` fix on 2026-11-10 follows current source:

- The graph starts at a `plan` node, not at `model` ([chat.py](../../services/engine/src/ledger_rag/orchestration/chat.py#L243-L251)).

The tool-error rule fires only above the threshold ([abstention.py](../../services/engine/src/ledger_rag/generation/abstention.py#L80-L81)).
An earlier "at or above" fix followed an uncommitted source change. That change was reverted, so the fix was reverted too.

ADR 0008 also lost one stale anchor link. The link it duplicated still points to the results section.

Measurement records stay unmapped: [EVALUATION.md](../EVALUATION.md), [DATASETS.md](../DATASETS.md) and [VALIDATION.md](../VALIDATION.md).

## Glossary, owners and known gaps

- [Glossary](glossary.md).
- **Owners:** unknown for every unit and page. Confirm with the project maintainer.
- **Known gaps:** no production topology is deployed. Quotas are per process. Voice is unverified on live calls ([release gates](../ARCHITECTURE.md#6-release-and-scaling-constraints)).
- **Review state:** an agent wrote every handbook page as a draft. No reviewer has approved any page.
  Mapped existing documents are `needs-review`. The chat journey page has no verified commit, because it describes uncommitted source.
