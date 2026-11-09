# ADR 0007: One corpus pipeline for every data source

- Status: accepted; core implemented 2026-11-09 (`ledger_rag.corpus`), live Modal
  build pending
- Date: 2026-11-09

## Context

Ingestion had grown around the FinQA benchmark: the Modal job special-cased
`source = "finqa"`, other datasets needed a manual `modal volume put`, the API's
`POST /v1/documents` was a third path, and nothing recorded which documents a
collection held, whether a build was complete or which index served traffic.
Production RAG platforms treat ingestion as a governed data pipeline: declarative
sources, data-quality gates, lineage, versioned indexes with atomic promotion,
permission-aware chunks that can be changed or erased without re-embedding, and
an audit trail. Benchmarks should be just one more source.

## Decision

`ledger_rag.corpus` runs every corpus through the same stages:

```text
config/corpora/<id>.toml ──► source ──► access policy ──► document gates
  ──► chunks ──► chunk gates ──► new collection <alias>__<build_id>
  ──► embed + write (lineage on every chunk) ──► readback gates
  ──► build record (status in Qdrant) ──► alias promotion / rollback
```

| Concern | Implementation |
|---|---|
| Declarative corpus | `CorpusSpec`: tenant, source, access policy, chunk profile, metadata fields, embedding space, serving alias, retention of builds |
| Sources | Pinned dataset registry (FinQA, TAT-QA, FinanceBench; ConvFinQA reuses FinQA) and JSONL `Document` files. New connectors only yield `Document`s |
| Access control | Tenant on every chunk; document ACL groups, with corpus defaults applied only to documents without an ACL (never widened); `confidential`/`restricted` corpora reject documents without an ACL; admin-only `PATCH /v1/documents/{id}/acl` and `DELETE /v1/documents/{id}` change or erase a document immediately without re-embedding, with hashed-id audit logs |
| Gates | Documents: identity, tenant, dates, ACL syntax, classification, near-empty, duplicate content. Chunks: lineage, contiguous ordinals, content hash, version, size. Index readback: total and per-document counts, payload contract and equality with the source, build id, date filter field, vector dimension/finiteness/non-zero. Reports hold codes and identifiers, never text |
| Lineage | `Document.metadata` and `content_version` (content only, so ACL changes do not invalidate it); chunks carry `metadata`, `document_version` and `build_id` |
| Status | `ledger_build` record in the collection metadata: `building` → `verified` → `promoted` / `superseded`, or `failed`, with counts, corpus fingerprint, gate summaries, git SHA and timing; Modal builds also write it to the `ledger-data` volume |
| Versioning | Every build is a new immutable collection; serving reads the alias; promotion is one atomic `update_collection_aliases`; rollback re-points to the previous verified build; pruning is explicit (`--yes`) |
| Compute | `python -m ledger_rag.corpus` (plan, build, status, promote, rollback, prune) locally; `modal run compute/ingestion.py::corpus --name <id>` builds on GPU with a private TEI |

Compatibility: documents without metadata keep their pre-existing fingerprint
(`Document.identity_json`), so evaluation can still verify the existing FinQA
collections. Planning the FinQA spec reproduces the recorded corpus exactly
(2,789 documents, 8,611 chunks). Existing collections can be adopted behind an
alias (`promote --collection <name> --adopt`) without rebuilding.

## Alternatives considered

| Option | Why not now |
|---|---|
| Keep dataset-specific jobs | Every new source would need another bespoke path without gates, lineage or promotion |
| Postgres document registry and outbox first | The right home for per-document status at scale and for multi-writer coordination, but not deployed; the build record plus per-chunk lineage covers single-writer builds today |
| Ingestion framework (LlamaIndex/Unstructured pipelines, Airflow/Dagster/Temporal) | Adds runtime and operational weight before there are connectors or schedules that need them; the stage contract here maps onto any of them later |
| Rebuild in place (upsert into the serving collection) | Readers would see half-built indexes and there would be no rollback |

## Production gaps (roadmap, in priority order)

1. **Incremental builds:** reuse vectors of unchanged documents (`document_version`)
   from the serving build and embed only changes; tombstones for deletions.
2. **Retrieval gate before promotion:** a small golden query set per corpus
   (recall@k) must not regress against the serving build.
3. **Document registry** (Postgres): per-document status, retries and a
   dead-letter queue, source ACL sync and audit history; the outbox from ADR-level
   gates in [architecture](../ARCHITECTURE.md#6-release-and-scaling-constraints).
4. **Connectors:** object storage and SEC EDGAR filings with parsing (PDF/HTML
   tables, OCR) and versioned parsers.
5. **Sensitive data:** PII detection and redaction before indexing; attribute-based
   access (clearance levels) beyond group ACLs; retention and legal hold.
6. **Freshness and observability:** schedules, freshness SLOs and per-stage metrics.

## Release gates

A first live `modal run compute/ingestion.py::corpus --name finqa` against the
workspace Qdrant; engine serving through an alias; the legacy `ingest` Modal
function and `IngestSpec` source special cases are removed afterwards.
