"""Normalize verified raw artifacts into ``documents.jsonl``, ``questions.jsonl`` and stats.

    uv run --frozen python -m ledger_rag.evaluation.datasets tatqa
    uv run --frozen python -m ledger_rag.evaluation.datasets convfinqa --finqa-dir data/finqa

Raw artifacts must already be fetched (``scripts/fetch_datasets.py``); their pinned
SHA-256 digests are re-verified before normalization.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ledger_rag.evaluation.datasets import convfinqa, financebench, tatqa
from ledger_rag.evaluation.datasets.fetch import sha256_file, verify
from ledger_rag.evaluation.datasets.schema import (
    CorpusDocument,
    QAItem,
    write_json,
    write_jsonl,
)
from ledger_rag.evaluation.datasets.sources import SOURCES
from ledger_rag.ingestion.chunking import chunk_document

ADAPTER_VERSION = 1


def load(dataset: str, raw_dir: Path) -> tuple[list[CorpusDocument], list[QAItem], Any]:
    if dataset == "convfinqa":
        raw = convfinqa.load_raw(raw_dir / "data.zip")
        documents, items = convfinqa.build(raw)
        return documents, items, raw
    if dataset == "tatqa":
        raw_tatqa = {
            split: json.loads((raw_dir / name).read_text(encoding="utf-8"))
            for split, name in tatqa.SPLIT_FILES.items()
        }
        documents, items = tatqa.build(raw_tatqa)
        return documents, items, raw_tatqa
    if dataset == "financebench":
        path = raw_dir / SOURCES["financebench"].artifacts[0].filename
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        documents, items = financebench.build(rows)
        return documents, items, rows
    raise ValueError(f"Unknown dataset {dataset}")


def statistics(documents: list[CorpusDocument], items: list[QAItem]) -> dict[str, Any]:
    splits: dict[str, Counter[str]] = {}
    for item in items:
        counter = splits.setdefault(item.split, Counter())
        counter["questions"] += 1
        counter[f"answer_type:{item.answer_type}"] += 1
        counter["multi_turn" if item.turns else "first_turn_or_single"] += 1
        counter["with_gold_units"] += bool(item.gold_unit_ids)
    units = Counter(unit.kind for document in documents for unit in document.units)
    return {
        "documents": len(documents),
        "chunks": sum(len(chunk_document(document.to_document())) for document in documents),
        "units": dict(units),
        "documents_with_fiscal_year": sum(d.fiscal_year is not None for d in documents),
        "companies": len({d.company for d in documents if d.company}),
        "questions": len(items),
        "splits": {split: dict(sorted(counter.items())) for split, counter in splits.items()},
        "tenants": sorted({d.tenant_id for d in documents}),
    }


def run(dataset: str, raw_dir: Path, out_dir: Path, finqa_dir: Path | None) -> dict[str, Any]:
    source = SOURCES[dataset]
    verify(raw_dir, dataset)
    documents, items, raw = load(dataset, raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "documents.jsonl", documents)
    write_jsonl(out_dir / "questions.jsonl", items)
    stats: dict[str, Any] = {
        "dataset": dataset,
        "adapter_version": ADAPTER_VERSION,
        "source": source.homepage,
        "revision": source.revision,
        "license": source.license,
        "usage": source.usage,
        "artifacts": {a.filename: a.sha256 for a in source.artifacts},
        "outputs": {
            name: sha256_file(out_dir / name) for name in ["documents.jsonl", "questions.jsonl"]
        },
        **statistics(documents, items),
    }
    if dataset == "convfinqa":
        stats["conversations"] = {split: len(examples) for split, examples in raw.items()}
        if finqa_dir is not None:
            from ledger_rag.evaluation.benchmarks.finqa import load as load_finqa

            corpus, _ = load_finqa(finqa_dir)
            stats["finqa_coverage"] = convfinqa.finqa_coverage(raw, corpus.documents)
    write_json(out_dir / "stats.json", stats)
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(SOURCES))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, help="default: <data-root>/<dataset>/normalized")
    parser.add_argument(
        "--finqa-dir", type=Path, help="FinQA data for ConvFinQA coverage (data/finqa)"
    )
    args = parser.parse_args(argv)
    raw_dir = args.data_root / args.dataset
    stats = run(args.dataset, raw_dir, args.out or raw_dir / "normalized", args.finqa_dir)
    print(json.dumps(stats, indent=2, sort_keys=True))
