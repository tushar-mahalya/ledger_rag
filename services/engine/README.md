# ledger-rag (engine)

The evidence-first financial QA API that every channel uses: authorization before
retrieval, hybrid search, bounded tools, citation and number validation, and explicit
abstention. Package `ledger_rag`; architecture in [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md).

| Package | Role |
|---|---|
| `api` | FastAPI app: query, SSE stream, documents, health, metrics, voice and MCP routes |
| `core` | Settings, shared models (the public contract), retry and deadline helpers |
| `orchestration` | `RagService` and the request-scoped LangGraph chat graph |
| `planning` | Query analysis, rewrite, decomposition and routing decisions |
| `retrieval` | Memory and Qdrant retrievers, reranking |
| `generation` | Providers, evidence-bound calculator, verification, abstention |
| `corpus` | Corpus pipeline: specs, sources, quality gates, versioned builds, alias promotion ([ADR 0007](../../docs/adr/0007-corpus-pipeline.md)) |
| `ingestion` | Chunking, contextual chunk profiles, TEI client, bulk writes |
| `external` | Point-in-time SEC EDGAR and FRED/ALFRED evidence |
| `mcp_adapter` | Opt-in authenticated MCP tools |
| `security`, `observability` | OIDC/demo auth and rate limits; logs, metrics, traces |
| `evaluation`, `training` | Benchmarks, scoring, uncertainty; embedding fine-tuning |
| `voice` | **Transitional** in-process gateway, replaced by `services/voice` ([ADR 0006](../../docs/adr/0006-voice-channel-service.md)) |

```bash
ledgerctl run -- uv run uvicorn ledger_rag.api.app:app --port 8080 --no-access-log
uv run pytest services/engine/tests
docker build -f services/engine/Dockerfile -t ledger-engine .   # from the repository root
```

Tests are split into `unit`, `integration`, `live` (opt-in, real SEC/FRED) and
`fixtures`.
