"""Fetch pinned, SHA-256-verified evaluation datasets into data/<dataset>/.

    uv run --frozen python scripts/fetch_datasets.py               # all datasets
    uv run --frozen python scripts/fetch_datasets.py tatqa convfinqa

FinanceBench is CC-BY-NC-4.0 and evaluation-only; never use it for training.
"""

import argparse
import sys
from pathlib import Path

from ledger_rag.evaluation.datasets.fetch import fetch
from ledger_rag.evaluation.datasets.sources import SOURCES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help=f"any of {sorted(SOURCES)} (default all)")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    if unknown := set(args.datasets) - set(SOURCES):
        parser.error(f"unknown datasets: {sorted(unknown)}")
    for name in args.datasets or sorted(SOURCES):
        source = SOURCES[name]
        downloaded = fetch(name, args.data_root / name)
        print(
            f"{name}: {source.license} ({source.usage}) @ {source.revision}; "
            f"verified {len(source.artifacts)} artifact(s), downloaded {len(downloaded)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
