---
name: ledger-compute
description: "Use when running or changing Ledger's Modal training, GPU embedding/reranking, dataset mining, ONNX/INT8 export, parity checks or model promotion."
---

# Right-sized Modal compute with auditable promotion

## Read

[Official Modal skill](../modal/SKILL.md),
[training app](../../../compute/training.py),
[inference app](../../../compute/inference.py),
[mining](../../../src/ledger_rag/training/data.py),
[registry](../../../src/ledger_rag/training/registry.py).

## Procedure

1. Match the official skill version to the locked SDK; use `--help` for actual
   signatures. Generic SDK examples do not override this project's safeguards.
2. Before remote calls, confirm account/profile, dataset rights, GPU availability,
   approved per-run spend and cancellation/timeout plan. Authenticate outside chat.
   Never reuse a pasted token or multiply accounts to bypass limits.
3. Run offline validation first:

   ```bash
   uv run --locked pytest -q tests/unit/test_training.py
   uv run --locked --group compute mypy --strict --ignore-missing-imports compute
   uv run --locked --group compute modal --version
   uv run --locked --group compute modal profile current
   ```

4. CPU handles fetch/mining and INT8 evaluation; GPU handles training and elastic
   batch inference. L4/L40S are inference options. Training currently defaults
   to H100 via `LEDGER_MODAL_TRAIN_GPU`; benchmark runtime x current price before
   calling that cheapest. Timeouts/container limits are not dollar caps.
5. Preserve locked images, Volume reload/commit boundaries, train/dev/test
   separation, false-negative filtering, production text rendering and clean
   Git lineage. Do not mount extra workspace files or secrets into images.
6. Only after approval, run one bounded job, observe it, and record its ID/cost:

   ```bash
   uv run --locked --group compute modal run compute/training.py --kind embedder --epochs 1
   ```

   Use `--kind reranker` for the separate reranker job; do not launch both blindly.
   Use [Vault projection](../../../scripts/sync_modal_secrets.sh) for serving keys.
   `modal deploy` creates persistent endpoints; it is not an offline check.
7. Inspect baseline/dev/guard metrics, held-out test, export status and INT8 parity.
   Current `PROMOTED` reflects the float-model dev gate, **not** an INT8 or live
   serving gate. Do not deploy it automatically. Dev checkpoint selection plus
   dev promotion is optimistic; test is reporting-only.
8. Verify fresh workers see committed artifacts. Version serving/index identities;
   check rollback before deploying. Record ONNX failure rather than claiming CPU
   parity. The current pipeline needs remote runtime validation before production.

## Done / stop

Return job/manifest identity, GPU/runtime/cost, metrics and deployment status.
Stop on missing auth, invalid data lineage, budget uncertainty, failed export,
non-finite metrics or unverified serving parity. Never train on held-out questions.
