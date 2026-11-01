---
name: ledger-evaluation
description: "Use when executing or comparing Ledger FinQA benchmarks, retrieval/quantization ablations, or model-promotion measurements. Not needed for ordinary retrieval code edits or general research."
---

# Ledger benchmark workflow

Commands and report interpretation: [evaluation runbook](../../../docs/EVALUATION.md).
Implementation: [FinQA](../../../src/ledger_rag/evaluation/benchmarks/finqa.py),
[generic harness](../../../src/ledger_rag/evaluation/harness.py).

## Run and compare

1. Freeze corpus/split, models, candidate counts, ACL scope and hardware before
   comparing arms. FinQA defaults to dev; test is sealed and needs `--unseal-test`
   (benchmark and `compute/training.py`) once the configuration is frozen.
2. Inspect existing jobs/results before ingestion. Verify pinned data checksums,
   completed ingestion and exact point counts before accepting a baseline.
3. Use identical cases for sparse/dense/hybrid/hybrid:rerank. Report company and
   global scopes separately. Run bulk model inference on Modal, not a local fallback.
4. Save machine-readable results with Git/data/model lineage, sample size,
   failures, latency and cost. Distinguish pilots from full-split measurements.
5. Quote `exact` (gold precision) as the headline and `tolerant` (1% relative)
   separately; `percent_scale_diagnostic` is never a headline. Quote report-cluster
   CIs, and use the paired differences for arm/model comparisons.

## Modal baseline (read-only against `finqa_gte_modernbert_v1`)

```bash
uv run --locked --group compute modal run compute/evaluation.py --command all \
  --limit 20 --answer-samples 20                       # smoke first
uv run --locked --group compute modal run compute/evaluation.py --command retrieval \
  --retrieval-concurrency 8                            # full dev, 4 arms x 2 scopes
uv run --locked --group compute modal run compute/evaluation.py --command answers \
  --models gemini-3.5-flash-lite --answer-samples 200 --llm-rpm 12 --llm-concurrency 4
uv run --locked --group compute modal volume get ledger-data evaluation/<run_id>.json .
```

- The GPU stage (L4 with L40S fallback; TEI `89-*` needs CC 8.9, A10G fails) serves
  the pinned embedder and reranker and saves answer contexts. The CPU stage calls the LLM;
  `--contexts-run <id>` reuses saved contexts with no GPU.
- Check `llm_calls.status_counts` and `error_samples`. A run with `http_429`
  failures is not a quality measurement: scored answers are the survivors. The free
  Gemini tier cannot run a dev baseline (run `448a606b`: `gemini-3.5-flash` quota
  of 20 requests/day/model; runs `44dbf8e0`, `728ae134`: 88% and 64% HTTP 429).

## Non-obvious harness limits

- FinQA `--limit` takes the first N selected-split cases; it does not limit ingestion.
  There is no ingest resume flag. Restarting re-embeds the full corpus.
- [Collection cloning](../../../scripts/clone_qdrant_benchmark.py) reuses vectors
  for quantization ablations. The source must be quiescent; the destination must
  not exist. Preserve the baseline and inspect partial destinations after failure.
- Qdrant scalar/binary compression and ONNX/INT8 model inference are different
  approximations. `none` bypasses vector quantization, including on compressed indexes.
- Answers are parsed from the final `ANSWER:` line only; format failures and
  abstentions count as wrong. This is not official program execution or semantic
  faithfulness. Citation precision/recall uses gold-unit-to-chunk mapping, not entailment.
- The generic harness requires `--remote --tenant` and explicit
  `--allow-index-write` for writes. Synthetic smoke is not a real-data quality score.

## Local checks (no model server needed)

```bash
uv run --locked --group compute pytest -q tests/unit/test_evaluation.py \
  tests/unit/test_finqa_benchmark.py tests/unit/test_evaluation_scoring.py \
  tests/unit/test_compute_evaluation.py
uv run --locked python -m ledger_rag.evaluation.harness --smoke --repeats 3
uv run --locked python -m ledger_rag.evaluation.benchmarks.finqa --help
```
