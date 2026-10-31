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
   comparing arms. Use FinQA `--split dev` for selection; test is reporting-only.
2. Inspect existing jobs/results before ingestion. Verify pinned data checksums,
   completed ingestion and exact point counts before accepting a baseline.
3. Use identical cases for sparse/dense/hybrid/hybrid:rerank. Report company and
   global scopes separately. Run bulk model inference on Modal, not a local fallback.
4. Save machine-readable results with Git/data/model lineage, sample size,
   failures, latency and cost. Distinguish pilots from full-split measurements.

## Non-obvious harness limits

- FinQA `--limit` takes the first N selected-split cases; it does not limit ingestion.
  There is no ingest resume flag. Restarting re-embeds the full corpus.
- [Collection cloning](../../../scripts/clone_qdrant_benchmark.py) reuses vectors
  for quantization ablations. The source must be quiescent; the destination must
  not exist. Preserve the baseline and inspect partial destinations after failure.
- Qdrant scalar/binary compression and ONNX/INT8 model inference are different
  approximations. `none` bypasses vector quantization, including on compressed indexes.
- Numeric answer matching is heuristic, not official program execution or
  semantic faithfulness. Citation validity is not entailment.
- The generic harness requires `--remote --tenant` and explicit
  `--allow-index-write` for writes. Synthetic smoke is not a real-data quality score.

## Local checks (no model server needed)

```bash
uv run --locked pytest -q tests/unit/test_evaluation.py tests/unit/test_finqa_benchmark.py
uv run --locked python -m ledger_rag.evaluation.harness --smoke --repeats 3
uv run --locked python -m ledger_rag.evaluation.benchmarks.finqa --help
```
