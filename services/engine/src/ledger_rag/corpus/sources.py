"""Source connectors: every source yields canonical ``Document`` records.

Benchmark datasets come through the pinned, SHA-256-verified dataset registry;
customer data comes through JSONL of ``Document`` records (one per line). New
connectors (object storage, SEC EDGAR filings, document stores) implement the same
generator contract, so validation, chunking, indexing and promotion stay shared.
"""

from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from ledger_rag.core.models import Document
from ledger_rag.corpus.spec import CorpusSpec
from ledger_rag.evaluation.datasets.build import NORMALIZED, run
from ledger_rag.evaluation.datasets.fetch import fetch_verified
from ledger_rag.evaluation.datasets.schema import CorpusDocument


class SourceError(ValueError):
    """The source cannot produce documents; never echoes document content."""


def _dataset_documents(spec: CorpusSpec, data_root: Path) -> Iterator[Document]:
    name = spec.source.name
    raw = data_root / name
    if name != "finqa" and name not in NORMALIZED:
        raise SourceError(f"Dataset {name!r} has no corpus adapter")
    fetch_verified(name, raw)
    if name == "finqa":
        from ledger_rag.evaluation.benchmarks.finqa import load

        corpus, _ = load(raw)
        yield from corpus.documents.values()
        return
    normalized = raw / "normalized"
    run(name, raw, normalized, None)  # deterministic; re-verifies the raw artifacts
    fields = set(spec.processing.metadata)
    with (normalized / "documents.jsonl").open() as stream:
        for number, line in enumerate(stream, 1):
            try:
                record = CorpusDocument.model_validate_json(line)
            except ValidationError:
                raise SourceError(f"{name}: invalid normalized document at line {number}") from None
            document = record.to_document()
            if fields:
                described = record.descriptive_metadata()
                extra = {key: value for key, value in described.items() if key in fields}
                document = document.model_copy(update={"metadata": {**document.metadata, **extra}})
            yield document


def _jsonl_documents(spec: CorpusSpec, data_root: Path) -> Iterator[Document]:
    path = (data_root / spec.source.path).resolve()
    if not path.is_relative_to(data_root.resolve()):
        raise SourceError("JSONL source must stay inside the data root")
    if not path.is_file():
        raise SourceError(f"JSONL source {spec.source.path} does not exist")
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield Document.model_validate_json(line)
            except ValidationError:
                raise SourceError(f"Invalid document at JSONL line {number}") from None


def load_documents(spec: CorpusSpec, data_root: Path) -> Iterator[Document]:
    if spec.source.kind == "dataset":
        return _dataset_documents(spec, data_root)
    return _jsonl_documents(spec, data_root)


def apply_access_policy(document: Document, spec: CorpusSpec) -> Document:
    """Give documents without an ACL the corpus default groups; never widen an ACL."""
    if document.allowed_groups or not spec.access.default_groups:
        return document
    return document.model_copy(update={"allowed_groups": list(spec.access.default_groups)})
