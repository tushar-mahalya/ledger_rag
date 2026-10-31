"""Bounded, complete-document ingestion into a newly claimed Qdrant collection."""

import hashlib
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ledger_rag.core.models import Chunk, Document
from ledger_rag.ingestion.chunking import chunk_document
from ledger_rag.retrieval.qdrant import QdrantRetriever, RetrievalError

log = structlog.get_logger(__name__)


class IngestSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    collection: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$")
    model_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    model_id: str = Field(
        default="Alibaba-NLP/gte-modernbert-base",
        pattern=r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$",
        max_length=180,
    )
    dimensions: int = Field(default=768, ge=32, le=8192)
    batch_size: int = Field(default=128, ge=1, le=512)
    tenant_id: str = Field(default="finqa", min_length=1, max_length=200)
    source: Literal["finqa", "jsonl"] = "finqa"
    input_path: str = ""
    quantization: Literal["none", "scalar", "binary"] = "none"

    @model_validator(mode="after")
    def validate_source(self) -> "IngestSpec":
        if self.source == "finqa" and (self.input_path or self.tenant_id != "finqa"):
            raise ValueError("FinQA uses the pinned dataset and the finqa tenant")
        if self.source == "jsonl":
            path = Path(self.input_path)
            if not path.is_absolute() or ".." in path.parts or not path.is_relative_to("/data"):
                raise ValueError("JSONL input must be an absolute file path inside /data")
        return self

    @property
    def embedding_space_id(self) -> str:
        return f"tei-1.8-float16:{self.model_id}@{self.model_revision}"


@dataclass(frozen=True)
class IngestResult:
    documents: int
    chunks: int
    batches: int
    corpus_sha256: str
    seconds: float


def jsonl_documents(path: Path) -> Iterator[Document]:
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                yield Document.model_validate_json(line)
            except ValidationError:
                raise ValueError(f"Invalid document at JSONL line {line_number}") from None


async def ingest_documents(
    documents: Iterable[Document],
    store: QdrantRetriever,
    *,
    tenant_id: str,
    batch_size: int = 128,
) -> IngestResult:
    """Write complete documents; the caller must own an empty, isolated collection."""
    if not 1 <= batch_size <= 512:
        raise ValueError("Ingestion batch size must be between 1 and 512")
    if await store.count() != 0:
        raise RetrievalError("Bulk ingestion requires an empty, isolated collection")
    start = time.monotonic()
    seen: set[str] = set()
    fingerprint = hashlib.sha256()
    batch: list[Chunk] = []
    chunks_written = batches = 0

    async def flush() -> None:
        nonlocal chunks_written, batches, batch
        await store.append_staging(batch)
        chunks_written += len(batch)
        batches += 1
        log.info("ingestion.batch_committed", batches=batches, chunks=chunks_written)
        batch = []

    for document in documents:
        if document.tenant_id != tenant_id:
            raise ValueError("Document tenant does not match the authorized ingestion tenant")
        if document.id in seen:
            raise ValueError("Duplicate document identity in ingestion input")
        seen.add(document.id)
        fingerprint.update(document.model_dump_json().encode())
        fingerprint.update(b"\n")
        chunks = chunk_document(document)
        if batch and len(batch) + len(chunks) > batch_size:
            await flush()
        batch.extend(chunks)
        # A large document stays whole for stale-chunk deletion; embedding and
        # Qdrant requests are independently sub-batched by the retriever.
        if len(batch) >= batch_size:
            await flush()
    if batch:
        await flush()
    if not seen:
        raise ValueError("Ingestion input contains no documents")
    actual = await store.count()
    if actual != chunks_written:
        raise RetrievalError(f"Ingestion count mismatch: expected {chunks_written}, found {actual}")
    result = IngestResult(
        documents=len(seen),
        chunks=chunks_written,
        batches=batches,
        corpus_sha256=fingerprint.hexdigest(),
        seconds=round(time.monotonic() - start, 3),
    )
    log.info("ingestion.verified", documents=result.documents, chunks=result.chunks)
    return result
