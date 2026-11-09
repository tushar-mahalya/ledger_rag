# Validation record

Latest refresh: **2026-11-01**. This record distinguishes measured behavior from
untested production targets. The current integrated checks and cloud experiments
are below; the older 2026-10-31 records are retained as history, not current counts.
Re-run the commands before relying on these results.

## Integrated source checks (2026-11-01)

Retrieval profiles, the reviewed bounded answer contract, open-model evaluation,
and external evidence/MCP features are merged.

- **1,096 Python tests passed, two opt-in live tests skipped**, independently on
  Python 3.11, 3.12 and 3.13. Service/helper and compute type checks passed on
  each interpreter. Remote-only compute imports use the existing
  `--ignore-missing-imports` compute check, including the session supervisor.
- Ruff lint and formatting passed; contract fixture parity and Agent Skills
  checks passed. The three-repeat synthetic smoke gate passed.
- **78 Vitest tests and 26 desktop/mobile Playwright cases passed**, including
  accessibility, grounding and external provenance. The real local API/browser
  flow passed with fictional data, no external model and no paid compute.
- Web lint, formatting, strict TypeScript, production build and budget checks
  passed: **175,576 JavaScript/CSS gzip bytes** against 204,800; largest
  minified JavaScript chunk **412,324 bytes** against 460,800.
- Wheel/source distribution builds and both Compose configurations passed.
  **18 Mermaid diagrams** parsed with the retained local validator; this is a
  syntax check, not visual or accessibility certification of diagrams.
- Seven independently reviewed correctness defects were corrected with
  regressions: intermediate calculation retention, ambiguous numeric units,
  per-share scaling, aggregate response size, FRED calendar-period selection,
  shared-fetch cancellation/shutdown, and unit-qualified SEC citation identity.
  Exact 900,000-byte UTF-8 response boundaries, escaping and explicit oversized
  SSE failure are tested rather than inferred from field-level schema parity.

## Bounded live checks (2026-11-01)

- Both rebuilt FinQA v2 collections had **8,647** payloads, chunk IDs and sparse
  vectors matching the regenerated checksum-pinned corpus and recorded builds.
  Legacy rendering metadata was corrected after that comparison; unrelated
  metadata was preserved. Readback and production v2 readiness passed; v1 was
  rejected. Dense vectors were not recomputed. This is compatibility evidence,
  not evidence that v2 improves answer quality.
- A bounded FRED metadata/observation check returned HTTP 200 with cutoff
  `2023-10-15`, observation date `2023-09-01`, and selected vintage `2023-10-02`.
  Citation identity, provenance and cutoff checks passed without retaining raw
  responses or keys. Broader live SEC/FRED coverage and production MCP clients
  remain release checks.

## Measured compute and remaining gates (2026-11-01)

[The evaluation record](EVALUATION.md) publishes pinned identities, report hashes,
sample counts, failures, confidence intervals, runtime and accounting:

- Prior full-dev retrieval run `c17d398f34f24efd874e29bb641d9738`.
- Serving benchmark `c0c06effb57e4581b86518d6c7e9803b` and 200-case pilot
  `4810c0ba57554f509f6301f388a80f35`.
- Selected full-dev runs `84cf8232546e457397d40f48e04b75ce` and
  `dcdfd41f28794c1182bc92be1ea77819`: **883 cases each**, tolerant numeric
  matching **61.27% retrieved@5 / 74.63% oracle**. Neither is official FinQA
  program-execution accuracy or a held-out/SOTA result.
- Both attached Apps stopped with zero tasks, runners and running inputs.
  Total elapsed time was **1,304.513 seconds**. Full-dev runs had zero provider
  failures/retries; the pilot retained one transport failure in its denominator.
- Combined per-App accrued usage **$1.74530226** differs from workspace metered
  usage **$2.07154255**. Conservative observed accounting is **$2.08**, not a
  settled invoice. The admission ledger remains reserved, with no additional
  elapsed admission; early finishes did not reset or refund it.

Still unverified: settled billing reconciliation, official comparable answer
scoring, held-out quality, trained-model quantization/serving parity, live carrier
interoperability, production warm serving and real voice latency. Workload-identity
wiring, abstention calibration and the company-scope retrieval latency anomaly
remain open. No production deployment or trained-model promotion occurred.

## Historical checks (2026-10-31)

## Verified source and packaging

```bash
uv run --locked --group compute ruff check .
uv run --locked --group compute ruff format --check .
uv run --locked --group compute mypy scripts/clone_qdrant_benchmark.py
uv run --locked --group compute mypy --strict --ignore-missing-imports compute
uv run --locked --group compute pytest -q
uv run --locked python scripts/check_agent_skills.py
uv build --no-sources
```

- 375 tests passed, including 52 new graph/history/stream regressions beyond the
  preceding 323-test baseline.
- Ruff lint passed; formatting checks passed for 91 files.
- Strict mypy passed for 45 service/helper files and three compute files.
- Wheel and source distribution built successfully.
- The Agent Skills validator passed, including canonical skills, local links,
  Claude aliases and the locked Modal skill version.
- All ten Mermaid diagrams parsed locally with Mermaid 11.12.2. This checks
  syntax only, not visual rendering.
- Warnings include intentional local execution of mocked Modal functions,
  SDK compatibility probes and local
  Qdrant's lack of payload indexing. Real-server indexing was checked in the
  earlier isolated-server tests below.
- Provider tests use mocked remote HTTP endpoints. Qdrant tests include the
  installed SDK's real local backend, not just response-shaped mocks.

## Chat workspace validation

Commands and architecture are in [the workspace guide](CHAT_WORKSPACE.md).

- **15 TypeScript unit tests** passed: SSE framing/UTF-8, runtime schemas,
  failure/completion bounds, history limits, safe references and artifact exports.
- **10 desktop/mobile Playwright cases** passed. They cover citation inspection,
  downloads, mocked live SSE/history, explicit failures, cancellation, token
  non-persistence, reduced motion, layout and axe WCAG A/AA checks. A strengthened
  cancellation case also passed on both viewports after explicitly switching
  conversations before releasing a delayed response.
- Automated accessibility checks caught low contrast and modal focus-restoration
  defects; both were fixed and re-tested. This is not a comprehensive manual
  screen-reader or all-browser accessibility certification.
- A separate **real local browser integration** passed through the same-origin
  proxy, authenticated API, LangGraph and citation gate over the fictional
  extractive provider. Actual activity and request/model/timing metadata rendered.
  Test API/UI processes were stopped; no external model, GPU or paid job ran.
- ESLint, Prettier, strict TypeScript and the Vite production build passed.
  Combined JavaScript/CSS gzip size was **162,954 bytes** (159.1 KiB), below the
  200 KiB gate. The largest minified JavaScript chunk was **381,364 bytes**,
  below 450 KiB. These are build measurements, not network or real-user SLOs.
- The pinned static Nginx image built and served as UID/GID 101, with a read-only
  filesystem, dropped capabilities and strict CSP. A browser rendered Markdown
  and opened the artifact inspector with **no page errors or CSP violations**.
  The temporary test container was stopped.
- Compose's chat overlay and the Collector configuration validated. The latter
  used a temporary non-secret credential fixture, not live telemetry credentials.
  Fixed graph stage/tool/profile values and outcome/count fields are retained;
  model strings and raw prompt/tool content are not allowlisted.
- Independent implementation review found no significant integration issues.
  No remote LLM accuracy, end-user latency, live Langfuse export, persistent
  conversation recovery or complete GenAI usage/cost reporting is claimed.

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
