---
name: ledger-operations
description: "Use when changing Docker deployment, Vault secrets, OIDC, health checks, logging, metrics, tracing, production configuration or incident recovery."
---

# Operate with explicit trust and failure boundaries

## Read

[Operations runbook](../../../docs/OPERATIONS.md),
[validation limits](../../../docs/VALIDATION.md),
[telemetry](../../../src/ledger_rag/observability/telemetry.py),
[settings](../../../src/ledger_rag/core/config.py),
[deployment assets](../../../deploy).

## Procedure

1. Establish target environment and resource ownership. This may be a shared
   machine: inspect named resources, never stop unrelated containers/processes.
2. Validate configuration without deploying:

   ```bash
   docker compose config --quiet
   docker compose -f compose.yaml -f deploy/compose.production.yaml --profile production config --quiet
   uv run --locked pytest -q tests/integration/test_api.py tests/integration/test_resilience.py
   ```

   The production overlay requires its documented environment values. Missing
   values should fail; example placeholders validate syntax, not readiness.

3. Require OIDC, persistent Qdrant, approved HTTPS providers and least-privilege
   Vault policies for production. Demo auth stays local. Restrict `/metrics`.
4. Secrets are files rendered by Vault Agent and read at startup; environment
   overrides can shadow rotated files. Remove stale overrides and roll instances.
   Never source templates, print values, copy them into images or log requests.
5. Preserve bounded-cardinality metrics, redacted structured logs and trace IDs.
   No tenant IDs, query text or document IDs as metric labels. Test failures and
   cancellation as well as successful traces.
6. Readiness must reflect usable dependencies; liveness must not cause cascading
   restarts during an upstream outage. Stop rollout on failed readiness.
7. For scaling, account for per-process quotas and ingestion serialization.
   Multiple workers do not create distributed admission control or atomic writes.
8. Before authorized deployment, record rollback/backup/restore steps, expected
   health and cost. Verify readiness and a scoped request; clean up only resources
   created for this task, with explicit names.

## Done / stop

Return validation output, health evidence and unresolved production gates.
Langfuse/GenAI semantics remain a proposed OTel integration, not a configured
destination. Do not export prompts or regulated data to an external sink without
approval. Container startup alone is not a successful deployment.
