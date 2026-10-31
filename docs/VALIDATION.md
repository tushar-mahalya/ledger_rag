# Validation record

Date: 2026-10-31. This record distinguishes measured local behavior from untested
production targets. Re-run the commands before relying on these results.

## Verified source and packaging

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/ledger_rag
uv run pytest -q
uv run python -m ledger_rag.evaluation.harness --smoke --repeats 20 --output reports/smoke.json
uv build
```

- 192 tests passed.
- Ruff lint and formatting passed.
- Strict mypy passed for 15 source modules.
- Wheel and source distribution built successfully.
- One upstream Starlette TestClient/httpx deprecation warning remains; it does not
  affect the passing assertions or the production HTTP client.
- Provider tests use mocked remote HTTP endpoints. Qdrant tests include the
  installed SDK's real local backend, not just response-shaped mocks.

## Measured synthetic benchmark

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

## Live HTTP smoke check

A local Uvicorn instance using a file-provisioned random demo credential verified:

- readiness returned 200;
- a voice request with publication cutoff `2025-12-31` returned the fictional
  FY2024 disclosure and no other-tenant evidence;
- SSE emitted a final answer and `done`, not an error;
- authenticated Prometheus metrics were available.

The temporary process was stopped and its temporary credential removed.

## Bounded concurrent HTTP measurements

Final source was also tested with four concurrent clients, two warmup requests
and 20 measured requests per profile, without retries:

| Local synthetic workload | Successes | Final-answer p95 | Evidence-event p95 |
|---|---:|---:|---:|
| Voice HTTP query | 20 / 20 | 18.512 ms | Not applicable |
| Chat staged SSE | 20 / 20 | 15.315 ms | 15.148 ms |

These short localhost, extractive-demo, closed-loop runs verify the load harness
and wire contracts, **not sustained capacity, production SLOs, semantic accuracy,
or real IVR latency**. Reports are generated under `reports/` and intentionally
excluded from Git. Every measured answer was non-abstaining. The temporary server
and credential were cleaned up.

## Container and secrets integration

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
inside the built container with the final workspace sources: they matched exactly.
This is local deployment evidence, not a completed public-cloud deployment.

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
