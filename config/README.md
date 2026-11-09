# config

Configuration shared by every unit. Nothing here is secret.

| Path | Contents |
|---|---|
| `secrets.toml` | The secret manifest: every credential's name, environments and delivery routes (values live in the encrypted store) |
| `corpora/<id>.toml` | Corpus specs: source, tenant, access policy, chunking, embedding space and serving alias for the corpus pipeline ([ADR 0007](../docs/adr/0007-corpus-pipeline.md)) |
| `env/dev.env`, `env/prod.env` | Non-secret `LEDGER_*` settings per environment, injected by `ledgerctl run` |

See [secrets and configuration](../docs/SECRETS.md).

## Adding your own data

Write one JSON `Document` per line (`id`, `tenant_id`, `title`, `text`, `source`,
`published_at`, optional `allowed_groups` and `metadata`) under `data/`, then add a
spec and run the gates before any build:

```toml
# config/corpora/acme_filings.toml
id = "acme_filings"
description = "ACME internal filings"
tenant = "acme"

[source]
kind = "jsonl"
path = "acme/filings.jsonl"     # relative to the data root

[access]
default_groups = ["finance"]    # for documents without their own ACL
classification = "confidential" # confidential/restricted demand an ACL on every document

[embedding]
model = "Alibaba-NLP/gte-modernbert-base"
revision = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
dimensions = 768

[index]
alias = "acme_filings_gte_modernbert"
```

```bash
ledgerctl run -- uv run python -m ledger_rag.corpus plan acme_filings
ledgerctl modal -- run compute/ingestion.py::corpus --name acme_filings --promote
```
