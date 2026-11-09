"""Data-quality gates for documents, chunks and the written index.

Errors block a build; warnings are recorded in the build report. Issues name the
document or chunk and a reason code, never the text itself, so reports are safe to
keep beside the index.
"""

import hashlib
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

from pydantic import ValidationError

from ledger_rag.core.models import Chunk, Document
from ledger_rag.corpus.spec import GROUP, CorpusSpec

Severity = Literal["error", "warning"]
MAX_ISSUES = 200
MIN_CHUNK_CHARS = 20
MAX_CHUNK_CHARS = 8000
EARLIEST = date(1900, 1, 1)


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: str
    subject: str


@dataclass
class Report:
    stage: str
    checked: int = 0
    issues: list[Issue] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)

    def add(self, severity: Severity, code: str, subject: str) -> None:
        self.counts[f"{severity}:{code}"] += 1
        if len(self.issues) < MAX_ISSUES:
            self.issues.append(Issue(severity, code, subject[:120]))

    @property
    def errors(self) -> int:
        return sum(n for key, n in self.counts.items() if key.startswith("error:"))

    @property
    def passed(self) -> bool:
        return self.errors == 0

    def summary(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "checked": self.checked,
            "passed": self.passed,
            "counts": dict(sorted(self.counts.items())),
            "examples": [issue.__dict__ for issue in self.issues[:20]],
        }


def validate_documents(documents: Sequence[Document], spec: CorpusSpec) -> Report:
    import re

    report = Report("documents", checked=len(documents))
    if not documents:
        report.add("error", "empty_corpus", spec.id)
    latest = (datetime.now(UTC) + timedelta(days=1)).date()
    seen: set[str] = set()
    versions: dict[str, str] = {}
    for document in documents:
        subject = document.id
        if document.id in seen:
            report.add("error", "duplicate_document_id", subject)
        seen.add(document.id)
        if document.tenant_id != spec.tenant:
            report.add("error", "tenant_mismatch", subject)
        if not EARLIEST <= document.published_at <= latest:
            report.add("error", "publication_date_out_of_range", subject)
        if any(not re.fullmatch(GROUP, group) for group in document.allowed_groups):
            report.add("error", "invalid_acl_group", subject)
        if spec.access.requires_acl and not document.allowed_groups:
            report.add("error", "missing_acl_for_classification", subject)
        if len(document.text.strip()) < MIN_CHUNK_CHARS:
            report.add("warning", "near_empty_document", subject)
        previous = versions.setdefault(document.content_version, document.id)
        if previous != document.id:
            report.add("warning", "duplicate_content", subject)
    return report


def validate_chunks(
    documents: Sequence[Document], chunks_by_document: Mapping[str, Sequence[Chunk]]
) -> Report:
    total = sum(len(chunks) for chunks in chunks_by_document.values())
    report = Report("chunks", checked=total)
    ids: set[str] = set()
    for document in documents:
        chunks = chunks_by_document.get(document.id, ())
        if not chunks:
            report.add("error", "document_without_chunks", document.id)
            continue
        if [chunk.ordinal for chunk in chunks] != list(range(len(chunks))):
            report.add("error", "non_contiguous_ordinals", document.id)
        for chunk in chunks:
            subject = f"{document.id}#{chunk.ordinal}"
            if chunk.id in ids:
                report.add("error", "duplicate_chunk_id", subject)
            ids.add(chunk.id)
            if (
                chunk.tenant_id != document.tenant_id
                or chunk.document_id != document.id
                or chunk.allowed_groups != document.allowed_groups
                or chunk.published_at != document.published_at
            ):
                report.add("error", "lineage_mismatch", subject)
            if chunk.content_hash != hashlib.sha256(chunk.text.encode()).hexdigest():
                report.add("error", "content_hash_mismatch", subject)
            if chunk.document_version != document.content_version:
                report.add("error", "document_version_mismatch", subject)
            if not chunk.text.strip():
                report.add("error", "empty_chunk", subject)
            elif len(chunk.text) > MAX_CHUNK_CHARS:
                report.add("error", "oversized_chunk", subject)
            elif len(chunk.text) < MIN_CHUNK_CHARS:
                report.add("warning", "tiny_chunk", subject)
    return report


def validate_vector(vector: Iterable[float], dimensions: int) -> str | None:
    """Reason code for an unusable dense vector, or None."""
    values = list(vector)
    if len(values) != dimensions:
        return "wrong_dimensions"
    if not all(math.isfinite(value) for value in values):
        return "non_finite_vector"
    if math.sqrt(sum(value * value for value in values)) < 1e-6:
        return "zero_vector"
    return None


def validate_readback(
    points: Sequence[tuple[Mapping[str, Any], Sequence[float] | None]],
    expected: Mapping[str, Chunk],
    *,
    dimensions: int,
    build_id: str,
    stored: int,
    counts_by_document: Mapping[str, tuple[int, int]],
) -> Report:
    """Check what the index actually holds against what the build wrote.

    ``points`` is a sample of stored payloads and dense vectors; ``expected`` maps chunk
    id to the chunk that was written; ``counts_by_document`` maps sampled document ids
    to (stored, expected) chunk counts.
    """
    report = Report("index", checked=len(points))
    if stored != len(expected):
        report.add("error", "point_count_mismatch", f"{stored}!={len(expected)}")
    for document_id, (actual, wanted) in counts_by_document.items():
        if actual != wanted:
            report.add("error", "document_chunk_count_mismatch", document_id)
    for payload, vector in points:
        try:
            chunk = Chunk.model_validate({k: v for k, v in payload.items() if k != "published_day"})
        except ValidationError:
            report.add("error", "payload_contract_violation", str(payload.get("id", "?"))[:16])
            continue
        subject = f"{chunk.document_id}#{chunk.ordinal}"
        source = expected.get(chunk.id)
        if source is None:
            report.add("error", "unexpected_point", subject)
            continue
        if chunk.model_dump(exclude={"build_id"}) != source.model_dump(exclude={"build_id"}):
            report.add("error", "payload_differs_from_source", subject)
        if chunk.build_id != build_id:
            report.add("error", "build_id_mismatch", subject)
        if payload.get("published_day") != chunk.published_at.toordinal():
            report.add("error", "date_filter_field_mismatch", subject)
        problem = "missing_vector" if vector is None else validate_vector(vector, dimensions)
        if problem:
            report.add("error", problem, subject)
    return report
