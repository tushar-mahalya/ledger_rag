"""Dataset pipeline CLI: fetch pinned sources, normalize them, prepare licensed splits.

    uv run python -m ledger_rag.evaluation.datasets fetch                  # every dataset
    uv run python -m ledger_rag.evaluation.datasets fetch finqa tatqa
    uv run python -m ledger_rag.evaluation.datasets build tatqa
    uv run python -m ledger_rag.evaluation.datasets build convfinqa --finqa-dir data/finqa
    uv run python -m ledger_rag.evaluation.datasets prepare-financebench --input ... \
        --corpus ... --mapping ... --output ... --accept-dataset-terms

Every download is pinned to an upstream revision and SHA-256 verified. FinanceBench is
CC-BY-NC-4.0 and evaluation-only; never use it for training. See docs/DATASETS.md.
"""

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from ledger_rag.evaluation.datasets.build import NORMALIZED, run
from ledger_rag.evaluation.datasets.fetch import fetch
from ledger_rag.evaluation.datasets.financebench_full import normalize
from ledger_rag.evaluation.datasets.sources import SOURCES


def _fetch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
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


def _build(args: argparse.Namespace) -> int:
    raw_dir = args.data_root / args.dataset
    stats = run(args.dataset, raw_dir, args.out or raw_dir / "normalized", args.finqa_dir)
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


def _prepare_financebench(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if not args.accept_dataset_terms:
        parser.error(
            "Review dataset and underlying disclosure rights, then explicitly accept terms"
        )
    try:
        rows, manifest = normalize(args.input, args.corpus, args.mapping)
    except (ValueError, OSError, ValidationError) as exc:
        parser.exit(
            1, f"Preparation failed ({type(exc).__name__}); check input files and manifest\n"
        )
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        parser.error("Output already exists; choose a new versioned path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(rows)} retrieval-only cases; no evidence or model answers were indexed.")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="python -m ledger_rag.evaluation.datasets", description=__doc__.split("\n\n")[0]
    )
    sub = root.add_subparsers(dest="command", required=True)
    fetch_p = sub.add_parser("fetch", help="download and verify pinned raw artifacts")
    fetch_p.add_argument("datasets", nargs="*", help=f"any of {sorted(SOURCES)} (default all)")
    fetch_p.add_argument("--data-root", type=Path, default=Path("data"))
    build_p = sub.add_parser("build", help="normalize a fetched dataset")
    build_p.add_argument("dataset", choices=NORMALIZED)
    build_p.add_argument("--data-root", type=Path, default=Path("data"))
    build_p.add_argument("--out", type=Path, help="default: <data-root>/<dataset>/normalized")
    build_p.add_argument("--finqa-dir", type=Path, help="FinQA data for ConvFinQA coverage")
    fb = sub.add_parser(
        "prepare-financebench", help="retrieval-only cases from a licensed full FinanceBench split"
    )
    fb.add_argument("--input", type=Path, required=True)
    fb.add_argument("--corpus", type=Path, required=True)
    fb.add_argument("--mapping", type=Path, required=True)
    fb.add_argument("--output", type=Path, required=True)
    fb.add_argument("--accept-dataset-terms", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    root = parser()
    args = root.parse_args(argv)
    if args.command == "fetch":
        return _fetch(args, root)
    if args.command == "build":
        return _build(args)
    return _prepare_financebench(args, root)
