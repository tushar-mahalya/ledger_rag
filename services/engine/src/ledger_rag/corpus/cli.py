"""Corpus pipeline CLI (``python -m ledger_rag.corpus``).

    plan     <corpus>                 gates only: source, access policy, documents, chunks
    build    <corpus> [--promote]     new build collection, readback gates, build record
    status   <corpus>                 serving build and every build's status
    promote  <corpus> <build_id> | --collection NAME [--adopt]
    rollback <corpus>                 serve the previous verified build
    prune    <corpus> --yes           delete non-serving builds beyond keep_builds

``build`` embeds through the configured endpoint (LEDGER_EMBEDDING_*); bulk GPU builds
run on Modal (``compute/ingestion.py``) with the same pipeline. Run through
``ledgerctl run`` so Qdrant and embedding credentials are present.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient

from ledger_rag.core.config import Settings
from ledger_rag.corpus.index import IndexRegistry, PromotionError
from ledger_rag.corpus.pipeline import BuildRejected, build, plan, store_settings
from ledger_rag.corpus.sources import SourceError, load_documents
from ledger_rag.corpus.spec import CorpusSpec, load_spec, new_build_id, spec_path
from ledger_rag.retrieval.qdrant import QdrantRetriever, RetrievalError


def _root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "config" / "corpora").is_dir():
            return candidate
    return Path.cwd()


def _registry(settings: Settings) -> IndexRegistry:
    key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return IndexRegistry(AsyncQdrantClient(url=settings.qdrant_url, api_key=key))


def _print(payload: Any) -> None:
    json.dump(payload, sys.stdout, indent=2, sort_keys=True, default=str)
    print()


async def _plan(spec: CorpusSpec, data_root: Path) -> int:
    planned = plan(spec, load_documents(spec, data_root))
    _print(
        {
            "corpus": spec.id,
            "documents": len(planned.documents),
            "chunks": planned.chunk_count,
            "passed": planned.passed,
            "reports": [report.summary() for report in planned.reports],
        }
    )
    return 0 if planned.passed else 1


async def _build(spec: CorpusSpec, data_root: Path, args: argparse.Namespace) -> int:
    from ledger_rag.core.config import get_settings

    planned = plan(spec, load_documents(spec, data_root))
    build_id = args.build_id or new_build_id()
    settings = store_settings(get_settings(), spec, spec.build_collection(build_id))
    store = QdrantRetriever(settings)
    registry = _registry(settings)
    try:
        record = await build(
            planned, store, build_id=build_id, registry=registry, promote=args.promote
        )
    except BuildRejected as exc:
        _print({"corpus": spec.id, "build_id": build_id, "error": str(exc), "reports": exc.reports})
        return 1
    finally:
        await store.close()
    _print(record)
    return 0


async def _admin(spec: CorpusSpec, args: argparse.Namespace) -> int:
    from ledger_rag.core.config import get_settings

    registry = _registry(get_settings())
    alias = spec.index.alias
    if args.command == "status":
        current = await registry.current(alias)
        builds = await registry.builds(alias)
        _print(
            {
                "corpus": spec.id,
                "alias": alias,
                "serving": current,
                "builds": [
                    {
                        "collection": b.collection,
                        "status": b.status,
                        "chunks": b.record.get("chunks"),
                        "finished_at": b.record.get("finished_at"),
                    }
                    for b in builds
                ],
            }
        )
    elif args.command == "promote":
        collection = args.collection or spec.build_collection(args.build_id)
        previous = await registry.promote(alias, collection, adopt=args.adopt)
        _print({"alias": alias, "serving": collection, "previous": previous})
    elif args.command == "rollback":
        _print({"alias": alias, "serving": await registry.rollback(alias)})
    elif args.command == "prune":
        if not args.yes:
            raise PromotionError("Pruning deletes collections; re-run with --yes")
        _print({"deleted": await registry.prune(alias, spec.index.keep_builds)})
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="python -m ledger_rag.corpus", description=__doc__.split("\n\n")[0]
    )
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("plan", "build", "status", "promote", "rollback", "prune"):
        command = sub.add_parser(name)
        command.add_argument("corpus", help="config/corpora/<corpus>.toml or a spec path")
        if name in {"plan", "build"}:
            command.add_argument("--data-root", type=Path, default=Path("data"))
        if name == "build":
            command.add_argument("--promote", action="store_true")
            command.add_argument("--build-id", default="")
        if name == "promote":
            command.add_argument("build_id", nargs="?", default="")
            command.add_argument("--collection", default="")
            command.add_argument(
                "--adopt", action="store_true", help="legacy index without a record"
            )
        if name == "prune":
            command.add_argument("--yes", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        spec = load_spec(spec_path(args.corpus, _root()))
        if args.command == "promote" and not (args.build_id or args.collection):
            raise PromotionError("Name a build id or --collection")
        if args.command == "plan":
            return asyncio.run(_plan(spec, args.data_root))
        if args.command == "build":
            return asyncio.run(_build(spec, args.data_root, args))
        return asyncio.run(_admin(spec, args))
    except (
        OSError,
        ValidationError,
        ValueError,
        SourceError,
        PromotionError,
        RetrievalError,
        RuntimeError,
    ) as exc:
        print(f"corpus: error: {type(exc).__name__}: {exc}"[:500], file=sys.stderr)
        return 1
