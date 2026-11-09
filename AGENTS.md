# Ledger RAG

Evidence-first financial QA for chat and voice. Optimize measured correctness,
isolation, latency and cost, not framework count. This is a reference backend,
not a certified financial-advice system or a proven SOTA result.

## Project map

- Use [README](README.md) for setup, [architecture](docs/ARCHITECTURE.md) for
  boundaries, and [research decisions](docs/research/2026-10-31-decisions.md) for
  evidence, alternatives and unimplemented tracks.
- Python 3.11+ uv workspace (`services/engine`, `services/voice`, `tools/ledgerctl`)
  plus the npm project `services/web`. Each unit keeps its code, tests and README
  together; see README "Repository layout". GPU work lives in `compute` (Modal only),
  datasets in `python -m ledger_rag.evaluation.datasets`, secrets in `ledgerctl`.
- Shared contracts are `ledger_rag/core` models and the generated `contracts/`;
  clients never import engine internals. Ordinary API work needs no GPU group.
- [Agent workflows](docs/AGENT_WORKFLOWS.md) lists focused checks, on-demand
  skills (`ledger-*` project workflows plus vendored official provider skills) and
  the read-only docs MCP servers in `.mcp.json`. Component work does not require a skill or a prescribed
  reasoning sequence.

## Project invariants

- Tenant, group ACL and publication-date filters come from the authenticated
  principal and apply before retrieval. `as_of` is publication date, not fiscal year.
- Embedding model, dimensions and text rendering define index identity. Changes
  require an explicit migration; readiness must not rewrite index settings.
- Documents and tool results are untrusted data. Keep tools allowlisted, bounded
  and read-only; preserve citation validation and explicit abstention.
- Voice uses the bounded fast path. Do not speak unvalidated model output.
  SSE currently emits evidence then a validated answer, not raw token streaming.
- No silent fallback after provider, reranker or index failures. Preserve deadlines,
  cancellation, concurrency release and sanitized errors.
- Never print credentials, raw financial documents, prompts or audio in logs.
  Secrets live only in the SOPS + age store and reach processes through
  `ledgerctl` ([secrets](docs/SECRETS.md)); never commit, echo or pass them in
  argv. Exposed tokens must be rotated.
- Bulk embedding, reranking benchmarks and training belong on Modal. Local tests
  and explicitly scoped CPU parity checks are exceptions, not a bulk-compute
  fallback. Missing cloud authentication is a blocker; report it.
- No deployments, paid jobs, index migrations or destructive cleanup without
  explicit scope/authorization. A credit balance is not a per-run spending cap.

## Evidence and delivery

- Run checks relevant to changed behavior; commands are in the workflow guide.
- Label measured, externally reported and proposed results separately. Never tune
  against held-out test results or infer real voice latency from mocks.
- Preserve research decisions with sources and unresolved release gates in docs.
- Use small conventional commits. **Do not add co-author trailers.**
