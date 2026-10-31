---
name: ledger-evaluation
description: "Use when running FinQA/FinanceBench benchmarks, retrieval ablations, numeric-answer evaluation, leakage checks, latency tests or model promotion evidence."
---

# Reproducible evaluation, not leaderboard theatre

## Read

[Methodology](../../../docs/EVALUATION.md),
[generic harness](../../../src/ledger_rag/evaluation/harness.py),
[FinQA harness](../../../src/ledger_rag/evaluation/benchmarks/finqa.py),
[training data](../../../src/ledger_rag/training/data.py).

## Procedure

1. State the hypothesis and primary/guard metrics before running. Fix corpus,
   split, seed, candidate counts, ACL scope, model revision, backend and hardware.
2. Check corpus rights and contamination. Train only on train; use dev for
   selection. Keep test untouched until the decision. Repeated test-driven tuning
   invalidates a held-out claim; use a new holdout.
3. Begin with the credential-free checks:

   ```bash
   uv run --locked pytest -q tests/unit/test_evaluation.py tests/unit/test_finqa_benchmark.py tests/unit/test_training.py
   uv run --locked python -m ledger_rag.evaluation.harness --smoke --repeats 3
   uv run --locked python -m ledger_rag.evaluation.benchmarks.finqa --help
   ```

4. For real data, verify the fetch script's pinned checksums, complete ingestion
   and expected corpus/chunk counts before scoring. A partially populated index
   is not a valid baseline. Inspect existing job/results before restarting work.
5. Compare sparse, dense, hybrid and hybrid:rerank with identical queries. FinQA
   supports `--configs`, `--scopes company global`, `--limit`, and
   `--answer-modes`; confirm accepted values in current CLI/source.
   Company-scoped and global retrieval are different tasks; report both.
6. Separate retrieved vs oracle evidence answers, numeric accuracy, abstention,
   citation validity and semantic support. A citation marker is not entailment.
7. Measure warm/cold latency, p50/p95/p99, concurrency, error rate and cost.
   Count failures in denominators. Report sample size, uncertainty, timeout/rate
   limits and whether the reranker consumed CPU or GPU.
8. Save machine-readable outputs with Git/data/model lineage; publish aggregates,
   not secrets or unrestricted source documents. Use the actual production path.

## Done / stop

Return baseline/candidate metrics, tradeoffs, artifact path and pass/fail decision.
Stop before remote evaluation without approved credentials, scope, cost and
mutation permission. The generic harness requires `--remote --tenant` and
`--allow-index-write` for writes. Do not pass real keys in CLI arguments.
Synthetic smoke success is never FinQA accuracy or real IVR latency.
