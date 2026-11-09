# data

| Path | Contents |
|---|---|
| `examples.jsonl` | Original, fictional disclosures seeding the keyless demo and tests |
| `eval.jsonl` | Synthetic smoke-evaluation cases for `ledger_rag.evaluation.harness --smoke` |
| `finqa/`, `convfinqa/`, `tatqa/`, `financebench/`, `raw/`, `index/` | Downloaded or derived data (`python -m ledger_rag.evaluation.datasets fetch` / `build`), ignored by git |

Real benchmark corpora keep their own licenses and are never committed; see
[datasets](../docs/DATASETS.md).
