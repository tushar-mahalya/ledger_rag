---
name: ledger-compute
description: "Use when executing or releasing Ledger's Modal training and GPU inference jobs, including ONNX/INT8 export and serving parity. Adds project release gates to the official Modal SDK skill; not for unrelated API work."
---

# Ledger Modal compute workflow

[training app](../../../compute/training.py),
[inference app](../../../compute/inference.py),
[colocated ingestion](../../../compute/ingestion.py),
[registry](../../../src/ledger_rag/training/registry.py).
SDK details: [upstream Modal skill](../modal/SKILL.md).
Credentials: [headless authentication](../../../docs/AGENT_WORKFLOWS.md#headless-modal-authentication).

## Execution boundary

- Bulk embeddings, reranking benchmarks and training run on Modal. Its CPU
  workers handle fetch/mining and INT8 evaluation; GPU workers handle models.
  The local CLI coordinates remote calls; it need not do model computation.
- Local unit tests, synthetic smoke and explicit CPU parity checks are valid.
  A warm query-serving tier avoids scale-to-zero cold starts on IVR traffic.
- Verify platform access with `modal app list --limit 5` under the locked compute
  environment. A profile selection alone is not proof of access. Missing auth
  blocks remote work; do not silently substitute local CPU inference.
- Platform tokens authenticate CLI/SDK calls. `ledger-tei` / `TEI_API_KEY` protects
  HTTP inference separately; use [Vault projection](../../../scripts/sync_modal_secrets.sh).

## Job and release checks

1. Confirm target workspace, data lineage, authorized resource/time limits and
   cancellation plan. H100 training and L4/L40S inference are defaults, not proven
   cost optima. Credit balances and timeouts are not hard spending caps.
2. Run the scoped job and retain its ID, artifact/manifest identity, runtime and
   cost. `modal deploy` publishes persistent endpoints, not an offline check.
3. Check committed artifacts from a fresh worker, export success and serving parity.
   `PROMOTED` currently gates only float-model dev metrics, not INT8 or live
   serving. Do not auto-deploy that pointer. Dev selection/gating is optimistic;
   held-out test is reporting-only. Remote runtime validation remains required.

## Offline validation

```bash
uv run --locked pytest -q tests/unit/test_training.py
uv run --locked --group compute mypy --strict --ignore-missing-imports compute
```
