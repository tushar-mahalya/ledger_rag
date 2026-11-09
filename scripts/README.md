# scripts

Repository tooling run from the root (`uv run python scripts/<name>.py`). Tests for
these live in `scripts/tests`. Product logic does not live here: the dataset pipeline
is `python -m ledger_rag.evaluation.datasets` ([datasets](../docs/DATASETS.md)) and
secrets are `ledgerctl`.

| Script | Area | Purpose |
|---|---|---|
| `build_gazetteer.py` | data | Ticker/alias gazetteer for query planning and voice keyterms |
| `run_eval_session.py` | evaluation | Supervised, budgeted Modal evaluation sessions |
| `benchmark_http.py` | evaluation | HTTP load benchmark against a running engine |
| `clone_qdrant_benchmark.py` | evaluation | Copy stored vectors into a new benchmark collection |
| `dump_contract_fixtures.py` | contract | Regenerate `contracts/` (and the web demo fixture); `--check` in CI |
| `export_requirements.py` | build | Hash-pinned Modal image requirements from `uv.lock`; `--check` in CI |
| `release_qdrant_write_lease.py` | operations | Release an abandoned Qdrant write lease after reconciliation |
| `check_agent_skills.py` | repository | Validate agent skills and their links |
