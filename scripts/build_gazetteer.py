"""Build the planner gazetteer (``LEDGER_PLANNER_GAZETTEER_PATH``) from corpus metadata.

FinQA pages are normalized with the same ``page_document`` used for indexing, so tickers come
from document ids such as ``ADI/2009/page_49.pdf``. Optional curated aliases
(``{"ADI": ["analog devices"]}``) are merged in; only their tickers present in the corpus are kept.

    uv run python scripts/build_gazetteer.py --finqa data/finqa --output data/gazetteer.json
    uv run python scripts/build_gazetteer.py --documents corpus.jsonl --output gazetteer.json
"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

from ledger_rag.core.models import Document
from ledger_rag.evaluation.benchmarks.finqa import page_document
from ledger_rag.ingestion.bulk import jsonl_documents
from ledger_rag.planning.analysis import Gazetteer

SPLITS = ("train", "dev", "test")


def finqa_documents(data_dir: Path) -> Iterator[Document]:
    found = [
        data_dir / f"{split}.json" for split in SPLITS if (data_dir / f"{split}.json").exists()
    ]
    if not found:
        raise FileNotFoundError(f"No FinQA split JSON files in {data_dir}")
    for path in found:
        for example in json.loads(path.read_text()):
            yield page_document(example)[0]


def build(
    *, finqa: Path | None = None, documents: Path | None = None, aliases: Path | None = None
) -> dict[str, list[str]]:
    corpus: list[Document] = []
    if finqa is not None:
        corpus.extend(finqa_documents(finqa))
    if documents is not None:
        corpus.extend(jsonl_documents(documents))
    gazetteer = Gazetteer.from_corpus(corpus)
    if aliases is not None:
        for ticker, names in Gazetteer.from_file(aliases).to_dict().items():
            if ticker in gazetteer.tickers:
                gazetteer.add(ticker, *names)
    return gazetteer.to_dict()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--finqa", type=Path, help="Directory with FinQA train/dev/test JSON")
    parser.add_argument("--documents", type=Path, help="Normalized Document JSONL")
    parser.add_argument("--aliases", type=Path, help="Curated {TICKER: [aliases]} JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.finqa is None and args.documents is None:
        parser.error("Provide --finqa and/or --documents")
    result = build(finqa=args.finqa, documents=args.documents, aliases=args.aliases)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(result)} tickers to {args.output}")


if __name__ == "__main__":
    main()
