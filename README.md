# Ledger RAG

**Evidence-first financial-disclosure intelligence, reusable across chat and voice.**

The problem: financial assistants confidently confuse reporting periods, misread
tables, fabricate citations, and leak documents across customers. Ledger separates
retrieval permissions, evidence provenance, bounded reasoning, and evaluation
instead of hiding them inside an unconstrained agent.

This is a working reference backend and a research-backed production architecture,
not a claim of absolute SOTA, financial-advice suitability, or guaranteed hiring.
The portfolio differentiator is reproducible evidence and engineering tradeoffs,
not the number of frameworks in the diagram.

## What actually runs

| Capability | Implementation |
|---|---|
| Reusable API | FastAPI query, ingestion, health, metrics, and SSE endpoints |
| Two execution profiles | Voice: at most three initial chunks, no tools/rerank; chat: bounded calculator and corrective retrieval |
| Provider portability | OpenAI chat-completions and Anthropic messages wire protocols, custom base URLs, normalized tool history |
| Local, keyless demo | Deterministic lexical retrieval and extractive responses over **fictional** disclosures |
| Persistent retrieval | Qdrant dense/sparse RRF retrieval with an OpenAI-compatible embedding endpoint |
| Reranking | Optional TEI cross-encoder over bounded authorized candidates; explicit failures, no silent bypass |
| Ingestion | Paragraph-aware bounded chunks, neighboring parent metadata, content hashes and stable tenant-scoped IDs |
| Identity and permissions | OIDC signature/issuer/audience/expiry validation; server-derived tenant and group ACLs, admin/reader roles |
| Evidence contract | Source, disclosure date and exact retrieved quote; unknown or missing paragraph citations trigger abstention |
| Tool harness | Allowlisted read-only retrieval and Decimal arithmetic; argument validation, cumulative tool budget, query deadline |
| Operations | Structured redacted logs, request/trace IDs, OTLP traces, Prometheus metrics, Docker, Vault Agent templates, CI |
| Evaluation | Retrieval/answer smoke tests, authorization/failure-path suites, reproducible evaluation and HTTP benchmarking |
| Voice gateway | Generic PCM and Twilio Media Streams adapters, VAD and barge-in; synthetic lifecycle tests, not verified carrier calls |
| Compute plane | Modal training/inference code, leakage-aware mining and model manifests; remote execution and serving promotion still need validation |
| Agent workflows | Portable Agent Skills, shared working agreement, official Modal reference and CI validation |

**Important boundaries:** valid citation markers do not establish semantic
entailment. The default demo is not neural RAG. SSE sends evidence followed by a
validated complete answer, **not** raw token streaming; this is deliberate to avoid
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

## Tests and evaluation

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/ledger_rag
uv run pytest -q
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
- [Primary-source research and dataset decisions](docs/RESEARCH.md)
- [Deployment, secrets, observability and recovery](docs/OPERATIONS.md)
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
| Dense embeddings | `LEDGER_EMBEDDING_BASE_URL`, model, matching dimensions and API key |
| Persistent index | `LEDGER_RETRIEVAL_BACKEND=qdrant`, URL/collection and optional API key |
| Dense index compression | `LEDGER_QDRANT_QUANTIZATION=none` (default), `scalar`, or experimental `binary`; `LEDGER_QDRANT_OVERSAMPLING=2.0` |
| Cross-encoder | Optional `LEDGER_RERANKER_URL` (TEI base URL) and API key |
| Production identity | OIDC issuer, audience, JWKS URL and signed `tenant_id`, `roles`, `groups` claims |
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
