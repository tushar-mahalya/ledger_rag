# Validation record

Date: 2026-10-31. This record distinguishes measured local behavior from untested
production targets. The current refresh covers source/packaging checks, local
FinQA statistics, documentation syntax and a read-only database preflight.
Earlier experiments are recorded separately, not claimed as reruns against the
current workspace. Re-run the commands before relying on these results.

## Verified source and packaging

```bash
uv run --locked --group compute ruff check .
uv run --locked --group compute ruff format --check .
uv run --locked --group compute mypy src/ledger_rag scripts/clone_qdrant_benchmark.py
uv run --locked --group compute mypy --strict --ignore-missing-imports compute
uv run --locked --group compute pytest -q
uv run --locked python scripts/check_agent_skills.py
uv build --no-sources
```

- 323 tests passed, including the FinQA split/reporting regressions.
- Ruff lint passed; formatting checks passed for 81 files.
- Strict mypy passed for 39 service/helper files and three compute files.
- Wheel and source distribution built successfully.
- The Agent Skills validator passed, including canonical skills, local links,
  Claude aliases and the locked Modal skill version.
- All nine Mermaid diagrams parsed locally with Mermaid 11.12.2. This checks
  syntax only, not visual rendering.
- Warnings include the upstream Starlette TestClient/httpx deprecation, intentional
  local execution of mocked Modal functions, SDK compatibility probes and local
  Qdrant's lack of payload indexing. Real-server indexing was checked in the
  earlier isolated-server tests below.
- Provider tests use mocked remote HTTP endpoints. Qdrant tests include the
  installed SDK's real local backend, not just response-shaped mocks.

## Local FinQA statistics

```bash
uv run --locked --group compute python -m ledger_rag.evaluation.benchmarks.finqa stats \
  --data data/finqa --split dev --limit 2
```

- The pinned local corpus produced 2,789 report pages and 8,611 chunks.
- The dev pool contains 883 questions; the bounded invocation selected two.
- This was a local statistics check only: no embeddings, model/cloud calls or
  database writes. The first two cases are not an accuracy benchmark, and
  `--limit` does not restrict the corpus.

## Read-only database preflight

- TLS-verified, authenticated `GET /` and `GET /collections` each returned
  HTTP 200 from the supplied Qdrant endpoint: server 1.20.0, zero collections.
  Neither the endpoint nor the credential is recorded here.
- No database writes, model runs or GPU jobs were performed. Read access does
  not establish collection-creation permissions, least-privilege write scope,
  backup/recovery readiness or reachability from a Modal worker.
- Modal read-only secret listing still returned no secrets; the Vault CLI was
  unavailable in the checked local environment. Approved Vault/Modal secret
  provisioning and an explicit paid-job budget remain blockers.
- Cloud GPU image execution, model loading, full-corpus ingestion, serving parity
  and throughput/cost remain **unverified**. Follow
  [the ingestion runbook](INGESTION.md) before launching a paid job.

## Earlier colocated ingestion validation

- Tests run the Modal function body with fake GPU/Volume dependencies and a real
  in-memory Qdrant store. They cover direct loopback embedding requests, complete
  document batching, bounded uploads, tenant/duplicate/input rejection, count
  mismatch, process exit/timeout/cancellation, failure sanitization and report
  persistence only after verified uploads.
- A remote-module-reimport regression verifies that the deployed query encoder
  retains its model/revision configuration; deployment-shell values alone do
  not propagate into Modal containers.
- Vault projection tests verify protected temporary files, no secret-bearing
  command arguments/logs, both projection targets, and cleanup on CLI failure.
- An earlier isolated live Qdrant 1.20.0 check uploaded three **synthetic** points,
  verified schema-3 identity across different embedding endpoint addresses, ACL/date
  filtering, existing-destination rejection and scalar vector copying. A further
  live check verified tenant-aware/document/group/date payload indexes in both
  original and copied staging collections. All smoke collections were removed;
  the partial real FinQA index was untouched. This was a separate local-server
  check, not a write test against the endpoint in the current read-only preflight.
- Production Compose accepts the external database/collection and embedding-space
  configuration. Modal CLI help loads the new job successfully. The synthetic
  evaluation's three-repeat threshold gate passed.

## Earlier measured synthetic benchmark

The following numerical tables retain the earlier recorded benchmark, not a new
neural or cloud measurement from the ingestion work.

Configuration: lexical memory backend, `extractive-demo`, chat profile, top-k 5,
four original fictional cases, 20 repeats (80 runs). Repeats are not additional
independent examples; confidence intervals cluster by case.

| Metric | Result |
|---|---:|
| Answerable-case document recall@5 | 1.000 |
| Answerable-case document MRR@5 | 1.000 |
| Answerable-case document nDCG@5 | 1.000 |
| Answerable-case document precision@5 | 0.200 |
| Citation-document relevance precision | 1.000 |
| Labeled substring checks | 1.000 |
| Abstention sensitivity / specificity | 1.000 / 1.000 |
| Answer coverage | 0.750 |

Precision@5 uses the requested five result slots as denominator, even when this
tiny corpus cannot supply five authorized documents. These metrics intentionally
remain visible rather than selecting only flattering scores.

Observed local, sequential retrieval p95 was approximately 0.083 ms and complete
demo execution p95 approximately 0.384 ms. **These are not voice, networked-model,
neural-retrieval, real-corpus or production latency measurements.** The report's
SHA256 manifest and configuration make the run reproducible; hardware contention
and warmup can change timing.

No real FinanceBench, FinQA, TAT-QA, or FinDER performance is claimed. There are no
semantic-faithfulness or numerical-correctness results without appropriate gold
labels. Degenerate intervals on four cases do not establish generalization.

## Earlier live HTTP smoke check

A local Uvicorn instance using a file-provisioned random demo credential verified:

- readiness returned 200;
- a voice request with publication cutoff `2025-12-31` returned the fictional
  FY2024 disclosure and no other-tenant evidence;
- SSE emitted a final answer and `done`, not an error;
- authenticated Prometheus metrics were available.

The temporary process was stopped and its temporary credential removed.

## Earlier bounded concurrent HTTP measurements

The then-current source was tested with four concurrent clients, two warmup
requests and 20 measured requests per profile, without retries:

| Local synthetic workload | Successes | Final-answer p95 | Evidence-event p95 |
|---|---:|---:|---:|
| Voice HTTP query | 20 / 20 | 18.512 ms | Not applicable |
| Chat staged SSE | 20 / 20 | 15.315 ms | 15.148 ms |

These short localhost, extractive-demo, closed-loop runs verify the load harness
and wire contracts, **not sustained capacity, production SLOs, semantic accuracy,
or real IVR latency**. Reports are generated under `reports/` and intentionally
excluded from Git. Every measured answer was non-abstaining. The temporary server
and credential were cleaned up.

## Earlier container and secrets integration

The operations implementation agent verified:

- image `ledger-rag:local` built and ran as UID 10001 without development
  dependencies;
- container live/ready checks, authenticated cited query, SSE, and protected
  metrics succeeded;
- default and production-overlay Compose configurations validated;
- Collector, Prometheus, Loki and Tempo configurations validated;
- a real local TLS Vault AppRole flow rendered secret templates that Pydantic
  consumed successfully;
- Gitleaks working-tree/history scans found no secrets;
- test containers, temporary volumes, credentials and certificates were removed.

An additional independent check compared SHA256 hashes of all 15 Python modules
inside the built container with the then-current workspace sources: they matched
exactly. This is local deployment evidence, not a completed public-cloud deployment.

## Review-driven fixes

The requested backend review identified two concrete defects:

1. A tiny negative-exponent arithmetic literal could expand into an enormous
   fixed-point string. Decimal exponents and formatted output are now bounded,
   with regression tests.
2. A whitespace-only document replacement could report success without replacing
   old evidence or permissions. Validation now rejects it before any batch
   mutation, with regression tests.

A later integrated benchmark also caught false abstention from an overly strict
single-year demo heuristic. The fix preserves whole comparative disclosure
sentences and their year/value attribution. The corpus was not changed to make
the test pass.

## Still required before public production

- Independently licensed/versioned full filings and held-out real benchmark runs.
- Actual LLM/embedding endpoints, hardware, model revisions, cost accounting and
  concurrent HTTP/voice latency tests.
- Production OIDC, Vault workload identity, TLS, target cloud/domain and private
  networking.
- Distributed indexing/versioned ACL coordination, persistent mutation recovery
  and gateway-wide quotas before multiple application replicas.
- Cloud deployment canaries, backup/restore drills, alert routing, semantic
  faithfulness audits and any telephony/STT/TTS integration.
