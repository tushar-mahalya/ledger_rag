---
name: ledger-retrieval
description: "Use when changing chunking, embeddings, BM25/dense fusion, Qdrant quantization, reranking, corpus identity or retrieval filtering in Ledger RAG."
---

# Retrieval changes without evidence or permission regressions

## Read

- [Chunking](../../../src/ledger_rag/ingestion/chunking.py),
  [Qdrant](../../../src/ledger_rag/retrieval/qdrant.py),
  [memory baseline](../../../src/ledger_rag/retrieval/memory.py),
  [reranking](../../../src/ledger_rag/retrieval/reranking.py).
- [Settings](../../../src/ledger_rag/core/config.py) and
  [research decisions](../../../docs/research/2026-10-31-decisions.md).

## Procedure

1. Define the failure with a representative table/prose/query and expected chunks,
   permissions, publication cutoff and top-k. Reproduce before changing ranking.
2. Preserve table headers, numeric units, source text and stable tenant-scoped IDs.
   Query prefixes belong only on queries. Training and serving must use the same
   `contextual_text` rendering and model/dimensions.
3. Keep tenant/ACL/as-of filters in every candidate branch before fusion/reranking,
   plus post-retrieval checks. Never retrieve globally and filter only afterward.
4. Treat model, dimensions, text rendering and endpoint identity changes as index
   migrations. Do not silently mix embedding spaces or rebuild a shared collection.
5. Quantization is opt-in: `none` bypasses compression; scalar/binary require a
   matching effective collection configuration. Preserve dense-only rescoring and
   oversampling. Readiness must not rewrite index settings.
6. Preserve ordered embedding sub-batches, index/shape/finite-value validation and
   explicit upstream errors. After uncertain writes, require reconciliation;
   in-process serialization is not a distributed transaction.
7. Test focused behavior, then use `ledger-evaluation` for frozen-corpus ablations.
   Binary compression and CPU INT8 model inference are different approximations;
   measure each separately.

```bash
uv run --locked pytest -q tests/unit/test_qdrant_store.py tests/unit/test_reranking.py tests/unit/test_core.py
```

## Done / stop

Report the regression test and ranking/latency tradeoff. Stop before migrations
without backup/rollback and authorization. Mocks and Qdrant local mode validate
contracts, not ANN recall, quantized index use or production throughput.
