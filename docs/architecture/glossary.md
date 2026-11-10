# Glossary

Terms used across this handbook, with the source that defines each one.

| Term | Meaning | Source |
|---|---|---|
| Abstention | An answer with no content, citations or calculations, plus client-safe reason codes | [abstention.py](../../services/engine/src/ledger_rag/generation/abstention.py) |
| Activity event | A safe SSE event that names a pipeline stage, tool and outcome; never reasoning text | [CHAT_API.md](../CHAT_API.md#activity-additive) |
| Admission slot | One permit of the engine's per-process query semaphore | [app.py](../../services/engine/src/ledger_rag/api/app.py#L173-L184) |
| ADR | A numbered architecture decision record in `docs/adr`, such as `0006`; the project's decision format | [docs/adr](../adr/) |
| `as_of` | Publication-date cutoff for evidence; not a fiscal year | [AGENTS.md](../../AGENTS.md) |
| Chunk | A tenant-owned piece of a document with groups, publication date and text | [models.py](../../services/engine/src/ledger_rag/core/models.py) |
| Chunk profile | The chunking and rendering version (`v1`, `v2`, `v3`); part of index identity | [ADR 0002](../adr/0002-chunk-profiles-and-qwen3-rerankers.md) |
| Citation | A numbered reference (`[C1]`) to one retrieved quote with source and date | [evidence.py](../../services/engine/src/ledger_rag/generation/evidence.py) |
| Evidence-bound calculator | A tool whose operands must be cited spans, earlier steps or small constants | [CHAT_API.md](../CHAT_API.md#grounded-calculation-and-verification) |
| Evidence event | The SSE event that lists retrieved sources before generation; not an answer | [evidence.py](../../services/engine/src/ledger_rag/generation/evidence.py#L58-L65) |
| Group ACL | `allowed_groups` on a chunk; an empty list allows every group in the tenant | [memory.py](../../services/engine/src/ledger_rag/retrieval/memory.py#L14-L19) |
| Hit | A chunk with a retrieval score | [models.py](../../services/engine/src/ledger_rag/core/models.py) |
| Index identity | Embedding model, dimensions and text rendering that bind a collection | [AGENTS.md](../../AGENTS.md), [ARCHITECTURE.md](../ARCHITECTURE.md#3-retrieval-and-index-contracts) |
| `ledgerctl` | Operator CLI that keeps secrets in SOPS + age and delivers them to runtimes | [SECRETS.md](../SECRETS.md) |
| Principal | The authenticated subject, tenant, roles and groups derived from the token | [auth.py](../../services/engine/src/ledger_rag/security/auth.py) |
| Profile | `chat` (tools, reranker, history) or `voice` (bounded fast path) | [service.py](../../services/engine/src/ledger_rag/orchestration/service.py#L78-L102) |
| Query plan | Request-scoped analysis, routing decision and sub-queries for retrieval | [ARCHITECTURE.md](../ARCHITECTURE.md#query-planning) |
| RRF | Reciprocal rank fusion of dense, sparse or sub-query result lists | [qdrant.py](../../services/engine/src/ledger_rag/retrieval/qdrant.py) |
| Tenant | The isolation boundary taken from the token's `tenant_id` claim | [auth.py](../../services/engine/src/ledger_rag/security/auth.py) |
| Tool budget | `LEDGER_MAX_TOOL_ROUNDS` tool calls per chat request, counted by call | [chat.py](../../services/engine/src/ledger_rag/orchestration/chat.py#L133-L135) |
| Verification mode | `off`, `flag` or `enforce`; `enforce` abstains on unsupported numbers | [verification.py](../../services/engine/src/ledger_rag/generation/verification.py#L287-L308) |
