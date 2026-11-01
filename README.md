# Ledger RAG

**Evidence-first financial-disclosure intelligence, reusable across chat and voice.**

The problem: financial assistants confidently confuse reporting periods, misread
tables, fabricate citations, and leak documents across customers. Ledger separates
retrieval permissions, evidence provenance, bounded reasoning, and evaluation
instead of hiding them inside an unconstrained agent.

This is a working reference backend and a research-backed production architecture,
not a claim of absolute SOTA or financial-advice suitability.
The portfolio differentiator is reproducible evidence and engineering tradeoffs,
not the number of frameworks in the diagram.

## What actually runs

| Capability | Implementation |
|---|---|
| Reusable API | FastAPI query, ingestion, health, metrics, and SSE endpoints |
| Research workspace | React/TypeScript chat, rich Markdown, source and artifact inspectors, real execution activity and per-request measurements |
| Two execution profiles | Voice: at most three initial chunks, no tools/rerank; chat: bounded calculator and corrective retrieval |
| Provider portability | OpenAI chat-completions and Anthropic messages wire protocols, custom base URLs, normalized tool history |
| Local, keyless demo | Deterministic lexical retrieval and extractive responses over **fictional** disclosures |
| Persistent retrieval | Qdrant dense/sparse RRF retrieval with an OpenAI-compatible embedding endpoint |
| Reranking | Optional TEI cross-encoder over bounded authorized candidates; explicit failures, no silent bypass |
| Ingestion | Paragraph-aware bounded chunks, neighboring parent metadata, content hashes and stable tenant-scoped IDs |
| Identity and permissions | OIDC signature/issuer/audience/expiry validation; server-derived tenant and group ACLs, admin/reader roles |
| Evidence contract | Source, disclosure date and exact retrieved quote; unknown or missing paragraph citations trigger abstention |
| Tool harness | Typed request-scoped LangGraph chat; allowlisted retrieval and Decimal arithmetic, bounded untrusted history, cumulative tool budget and deadlines |
| Operations | Structured redacted logs, request/trace IDs, OTLP traces, Prometheus metrics, Docker, Vault Agent templates, CI |
| Evaluation | Retrieval/answer smoke tests, authorization/failure-path suites, reproducible evaluation and HTTP benchmarking |
| Voice gateway | Generic PCM and Twilio Media Streams adapters, VAD and barge-in; synthetic lifecycle tests, not verified carrier calls |
| Compute plane | Colocated Modal GPU ingestion with direct Qdrant writes, training/inference, mining and model manifests; remote execution still needs validation |
| Agent workflows | Portable Agent Skills, shared working agreement, official Modal reference and CI validation |

**Important boundaries:** valid citation markers do not establish semantic
entailment. The default demo is not neural RAG. SSE sends evidence followed by a
validated complete answer, **not** raw token streaming; safe execution activity is
streamed separately. This is deliberate to avoid
speaking an answer before the citation gate. Real telephony calls, signed webhook
token minting and end-to-end voice latency remain unverified.
GraphRAG, ColPali, late interaction, validated GPU serving, durable workflow orchestration,
Postgres outbox, distributed quotas, and semantic faithfulness judges are
architecture/experiment tracks, not pretend implementations.

## Start locally

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/), optionally Docker.
Commands are run from this repository:

```bash
uv sync --frozen
export LEDGER_DEMO_TOKEN="$(openssl rand -hex 32)"
export LEDGER_SEED_PATH=data/examples.jsonl
uv run uvicorn ledger_rag.api.app:app --host 127.0.0.1 --port 8080 --no-access-log
```

In another terminal, set the same token securely and call:

```bash
curl -sS http://127.0.0.1:8080/health/ready
curl -sS http://127.0.0.1:8080/v1/query \
  -H "Authorization: Bearer $LEDGER_DEMO_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question":"What was Aurora revenue in fiscal year 2024?","as_of":"2025-12-31","profile":"voice"}'
curl -N http://127.0.0.1:8080/v1/query/stream \
  -H "Authorization: Bearer $LEDGER_DEMO_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question":"What was Aurora revenue in fiscal year 2025?","profile":"chat"}'
```

`as_of` filters the **disclosure publication date**, not the fiscal year mentioned
in the question. Client metadata never grants tenant or group membership.

### Docker

```bash
export LEDGER_DEMO_TOKEN="$(openssl rand -hex 32)"
docker compose up --build
```

The default binds to localhost and uses fictional demo data. For optional
infrastructure, Vault bootstrap, telemetry, and the production overlay see
[the operations runbook](docs/OPERATIONS.md). Do not expose demo authentication
publicly or treat an unsecured local Vault as a production secrets service.

### Chat workspace

With Node 22.12+:

```bash
npm ci --prefix web --ignore-scripts
npm --prefix web run dev
```

Open `http://127.0.0.1:5173` for the clearly labeled fictional design preview.
Start the API above and use **Connect API** with your Ledger access token for
live authenticated queries. Provider keys stay on the backend; the browser
never receives them. Conversations and access tokens are tab-memory-only.
See [workspace design, deployment and checks](docs/CHAT_WORKSPACE.md) and
[the graph/history/SSE contract](docs/CHAT_API.md).

## Tests and evaluation

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/ledger_rag
uv run --group compute pytest -q
uv run python -m ledger_rag.evaluation.harness --smoke --repeats 20 --output reports/smoke.json
```

The included data is original, tiny, synthetic test data. Its scores measure smoke
correctness only, **not performance on FinanceBench or real financial filings**.
See [evaluation methodology](docs/EVALUATION.md) for benchmark protocols,
confidence intervals, leakage controls, real dataset licensing, and load tests.
Real accuracy and voice latency must be measured with your endpoint, corpus,
hardware and concurrency. A model call can dominate the entire voice budget.

Tests cover cross-tenant and group isolation, publication cutoffs, stale chunk
replacement, malformed citations, unsafe arithmetic, OIDC claims, request-size
limits, quota exhaustion, upstream failures, deadlines, SSE errors, and release
of concurrency capacity.

## Architecture and research

- [Coding-agent workflows and portable skills](docs/AGENT_WORKFLOWS.md)
- [Current decisions, tradeoffs and open experiments](docs/research/2026-10-31-decisions.md)
- [Architecture and execution paths](docs/ARCHITECTURE.md)
- [Chat workspace, design research and browser validation](docs/CHAT_WORKSPACE.md)
- [Primary-source research and dataset decisions](docs/RESEARCH.md)
- [Deployment, secrets, observability and recovery](docs/OPERATIONS.md)
- [Single-container Modal ingestion and database uploads](docs/INGESTION.md)
- [Evaluation and experiment design](docs/EVALUATION.md)
- [Measured validation results and remaining promotion gates](docs/VALIDATION.md)

Start with a modular monolith for the synchronous query path. Extract indexing,
model serving, and voice adapters when their scaling/failure domains require it.
Multiple network hops and agent rounds are not a virtue in a latency-sensitive
request. See the architecture's promotion gates before introducing graph
retrieval or multimodal indexes.

## Configure real models

Non-secret configuration comes from `LEDGER_*` environment variables. Secrets can
come from Vault Agent-rendered, read-only files in `LEDGER_SECRETS_DIR`.
The application never implicitly loads a dotenv file or sources shell templates.
Explicit environment values override secret files; remove stale secret env values.
Secrets are read at startup: rotate files, then roll the application instances.

| Needed | Configuration |
|---|---|
| OpenAI-compatible LLM | `LEDGER_PROVIDER=openai`, `LEDGER_LLM_BASE_URL`, `LEDGER_LLM_MODEL`, `LEDGER_LLM_API_KEY` |
| Anthropic-compatible LLM | `LEDGER_PROVIDER=anthropic`, messages-compatible base URL, model and key |
| Generation controls | Optional `LEDGER_LLM_TEMPERATURE`, `LEDGER_LLM_MAX_TOKENS` (sent as `LEDGER_LLM_MAX_TOKENS_PARAMETER`: `max_tokens` or `max_completion_tokens`), `LEDGER_LLM_SEED`, `LEDGER_LLM_REASONING_EFFORT` (`none`…`high`, e.g. Gemini's OpenAI layer); unset values are not sent. Anthropic sends `max_tokens` (default 2048) and temperature ≤1, and refuses seed/reasoning effort at startup |
| Dense embeddings | `LEDGER_EMBEDDING_BASE_URL`, model, matching dimensions and API key |
| Colocated ingestion identity | `LEDGER_EMBEDDING_SPACE_ID` from the verified job manifest; empty retains legacy endpoint-bound collections |
| Persistent index | `LEDGER_RETRIEVAL_BACKEND=qdrant`, URL/collection and optional API key; `LEDGER_QDRANT_BOOTSTRAP_ON_STARTUP=true` provisions a missing collection (with payload indexes) at startup, because readiness never creates one |
| Dense index compression | `LEDGER_QDRANT_QUANTIZATION=none` (default), `scalar`, or experimental `binary`; `LEDGER_QDRANT_OVERSAMPLING=2.0` |
| Answer verification | `LEDGER_VERIFICATION_MODE=enforce` (default), `flag` or `off`; optional `LEDGER_GROUNDEDNESS_VERIFIER=lettucedetect` with URL/key (off by default, fails open) |
| Cross-encoder | Optional `LEDGER_RERANKER_URL` (TEI base URL) and API key |
| Upstream retries | `LEDGER_UPSTREAM_MAX_RETRIES` (default 2; voice is capped at 1), `LEDGER_UPSTREAM_RETRY_BASE_SECONDS`, `LEDGER_UPSTREAM_RETRY_MAX_DELAY_SECONDS`: full-jitter, `Retry-After`-aware, deadline-bounded retries of LLM/embedding/reranker calls ([operations](docs/OPERATIONS.md#health-capacity-and-failure-handling)) |
| Production identity | OIDC issuer, audience, JWKS URL and signed `tenant_id`, `roles`, `groups` claims; `sub` ≤256 and `tenant_id` ≤200 chars, ≤32 roles, ≤128 groups, each ≤256 chars, else 401 (duplicates are dropped; rejections log field names only) |
| Vault | Vault address, workload AppRole bootstrap or production workload identity, least-privilege policy |
| Deployment | Target cloud/region/domain, TLS certificate, budget, CPU/GPU and model/data residency requirements |
| IVR integration | Telephony credentials, signed-webhook/token-minting integration, STT/TTS endpoints and language/turn-taking requirements |

Any model behind the supported **wire protocols** can be configured. Actual tool
calling depends on that model/server's support; an arbitrary HTTP endpoint is not
automatically compatible. The adapter does not implement Responses-only APIs.
Never paste credentials into chat or commit them; provision them in Vault.

## Production promotion checklist

1. License and version the real corpus; freeze untouched benchmark splits.
2. Measure dense/sparse/hybrid/reranked ablations, numerical correctness,
   unsupported-answer rate, and p95/p99 under sustained concurrency.
3. Set up OIDC, gateway-wide quotas, TLS, private service networking and egress
   allowlists; restrict `/metrics` to a monitoring identity.
4. Make indexing transactional/versioned before concurrent ingestion, retention,
   hard deletions, or authorization changes are served at scale.
5. Provision Vault securely; exercise credential rotation, Qdrant recovery,
   outage/cancellation tests and telemetry retention.
6. Deploy to your selected cloud; execute real-model canaries and latency/quality
   gates before accepting external traffic.

### License

Project source is MIT-licensed. Models, datasets, third-party services and filing
content retain their own licenses and terms. No real benchmark corpus or model
weights are redistributed here.
