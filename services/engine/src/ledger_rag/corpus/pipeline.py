"""Plan and build a corpus: gates, a new build collection, readback and the build record."""

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

from ledger_rag.core.config import Settings
from ledger_rag.core.models import Chunk, Document
from ledger_rag.corpus.index import IndexRegistry
from ledger_rag.corpus.sources import apply_access_policy
from ledger_rag.corpus.spec import CorpusSpec
from ledger_rag.corpus.validate import (
    Report,
    validate_chunks,
    validate_documents,
    validate_readback,
)
from ledger_rag.ingestion.bulk import ingest_documents
from ledger_rag.ingestion.chunking import chunk_document
from ledger_rag.ingestion.context import document_hash
from ledger_rag.retrieval.qdrant import QdrantRetriever

log = structlog.get_logger(__name__)
COUNTED_DOCUMENTS = 25


class BuildRejected(RuntimeError):
    """A gate failed; ``reports`` explain why without document content."""

    def __init__(self, message: str, reports: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.reports = reports


@dataclass
class Plan:
    spec: CorpusSpec
    documents: list[Document]
    chunks: dict[str, list[Chunk]]
    reports: list[Report] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(report.passed for report in self.reports)

    @property
    def chunk_count(self) -> int:
        return sum(len(chunks) for chunks in self.chunks.values())


def plan(
    spec: CorpusSpec,
    documents: Iterable[Document],
    *,
    summaries: Mapping[str, str] | None = None,
) -> Plan:
    """Apply the access policy, chunk and run the document and chunk gates (no I/O)."""
    governed = [apply_access_policy(document, spec) for document in documents]
    reports = [validate_documents(governed, spec)]
    chunks: dict[str, list[Chunk]] = {}
    for document in governed:
        summary = None
        if spec.processing.chunk_profile == "v3":
            summary = (summaries or {}).get(document_hash(document.title, document.text))
        try:
            chunks[document.id] = chunk_document(
                document, profile=spec.processing.chunk_profile, summary=summary
            )
        except ValueError:
            chunks[document.id] = []
    reports.append(validate_chunks(governed, chunks))
    return Plan(spec, governed, chunks, reports)


def store_settings(base: Settings, spec: CorpusSpec, collection: str) -> Settings:
    """Engine settings for writing one build collection of ``spec``."""
    return base.model_copy(
        update={
            "retrieval_backend": "qdrant",
            "qdrant_collection": collection,
            "qdrant_quantization": spec.embedding.quantization,
            "embedding_model": spec.embedding.model,
            "embedding_dimensions": spec.embedding.dimensions,
            "embedding_space_id": spec.embedding.space_id,
            "chunk_profile": spec.processing.chunk_profile,
        }
    )


async def build(
    planned: Plan,
    store: QdrantRetriever,
    *,
    build_id: str,
    registry: IndexRegistry | None = None,
    promote: bool = False,
    batch_size: int = 128,
    sample_size: int = 64,
    summaries: Mapping[str, str] | None = None,
    git_sha: str = "",
) -> dict[str, Any]:
    """Write ``planned`` into the store's new, empty collection and verify it.

    Returns the build record, which is also stored in the collection metadata. The
    serving alias moves only when ``promote`` is set and every gate passed.
    """
    spec = planned.spec
    if not planned.passed:
        raise BuildRejected(
            f"{spec.id}: input gates failed", [r.summary() for r in planned.reports]
        )
    started = datetime.now(UTC)
    clock = time.monotonic()
    record: dict[str, Any] = {
        "corpus": spec.id,
        "build_id": build_id,
        "status": "building",
        "tenant": spec.tenant,
        "classification": spec.access.classification,
        "source": spec.source.model_dump(exclude_defaults=True),
        "embedding_space_id": spec.embedding.space_id,
        "chunk_profile": spec.processing.chunk_profile,
        "git_sha": git_sha,
        "started_at": started.isoformat(timespec="seconds"),
    }
    await store.create_empty_collection()
    await store.set_build_record(record)
    log.info("corpus.build_started", corpus=spec.id, build_id=build_id)
    try:
        result = await ingest_documents(
            planned.documents,
            store,
            tenant_id=spec.tenant,
            batch_size=batch_size,
            chunk_profile=spec.processing.chunk_profile,
            summaries=summaries,
            build_id=build_id,
        )
        expected = {
            chunk.id: chunk.model_copy(update={"build_id": build_id})
            for chunks in planned.chunks.values()
            for chunk in chunks
        }
        counted = planned.documents[:: max(1, len(planned.documents) // COUNTED_DOCUMENTS)]
        counts = {
            document.id: (
                await store.count_document(spec.tenant, document.id),
                len(planned.chunks[document.id]),
            )
            for document in counted[:COUNTED_DOCUMENTS]
        }
        readback = validate_readback(
            await store.sample_points(sample_size),
            expected,
            dimensions=spec.embedding.dimensions,
            build_id=build_id,
            stored=await store.count(),
            counts_by_document=counts,
        )
    except BaseException:
        await _record_failure(store, record, [r.summary() for r in planned.reports])
        raise
    reports = [r.summary() for r in [*planned.reports, readback]]
    record.update(
        documents=result.documents,
        chunks=result.chunks,
        corpus_sha256=result.corpus_sha256,
        reports=reports,
        finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
        seconds=round(time.monotonic() - clock, 3),
    )
    if not readback.passed:
        record["status"] = "failed"
        await store.set_build_record(record)
        raise BuildRejected(f"{spec.id}: index readback failed", reports)
    record["status"] = "verified"
    await store.set_build_record(record)
    log.info("corpus.build_verified", corpus=spec.id, build_id=build_id, chunks=result.chunks)
    if promote:
        if registry is None:
            raise ValueError("Promotion needs an IndexRegistry")
        previous = await registry.promote(spec.index.alias, spec.build_collection(build_id))
        record.update(status="promoted", previous=previous)
    return record


async def _record_failure(
    store: QdrantRetriever, record: dict[str, Any], reports: list[dict[str, Any]]
) -> None:
    try:
        await store.set_build_record({**record, "status": "failed", "reports": reports})
    except Exception:  # the original failure matters more than the status write
        log.warning("corpus.failure_record_unwritten", build_id=record.get("build_id"))
