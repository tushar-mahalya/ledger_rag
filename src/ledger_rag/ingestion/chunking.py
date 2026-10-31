"""Structure-aware chunking for financial disclosures.

Paragraphs and tables are kept intact where possible. Oversized markdown tables are
split on row boundaries and every continuation repeats the header rows, so a
retrieved fragment never contains numbers without their column labels. Oversized
prose is split on sentence boundaries before falling back to overlapping windows.
"""

import hashlib
import re

from ledger_rag.core.models import Chunk, Document

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _is_table(section: str) -> bool:
    lines = section.splitlines()
    return len(lines) >= 2 and all(line.lstrip().startswith("|") for line in lines)


def _split_table(section: str, max_chars: int) -> list[str]:
    lines = section.splitlines()
    header_count = 2 if len(lines) > 1 and re.fullmatch(r"\s*\|[\s:|-]+\|\s*", lines[1]) else 1
    header = "\n".join(lines[:header_count])
    if len(header) > max_chars // 2:
        return _windows(section, max_chars, 0)
    parts: list[str] = []
    current: list[str] = []
    size = len(header)
    for row in lines[header_count:]:
        if len(row) + len(header) + 1 > max_chars:
            if current:
                parts.append("\n".join([header, *current]))
                current, size = [], len(header)
            parts.extend(
                header + "\n" + piece for piece in _windows(row, max_chars - len(header) - 1, 0)
            )
            continue
        if current and size + len(row) + 1 > max_chars:
            parts.append("\n".join([header, *current]))
            current, size = [], len(header)
        current.append(row)
        size += len(row) + 1
    if current:
        parts.append("\n".join([header, *current]))
    return parts or [header]


def _windows(text: str, max_chars: int, overlap: int) -> list[str]:
    step = max(1, max_chars - overlap)
    return [text[i : i + max_chars] for i in range(0, len(text), step)]


def _split_prose(section: str, max_chars: int, overlap: int) -> list[str]:
    parts: list[str] = []
    current = ""
    for sentence in _SENTENCE.split(section):
        if len(sentence) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.extend(_windows(sentence, max_chars, overlap))
        elif current and len(current) + len(sentence) + 1 > max_chars:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        parts.append(current)
    return parts


def contextual_text(chunk: Chunk) -> str:
    """Text used for embedding, sparse indexing and reranking (contextual chunk header)."""
    return f"{chunk.title}\n{chunk.text}"


def chunk_document(document: Document, max_chars: int = 1800, overlap: int = 200) -> list[Chunk]:
    """Keep paragraphs/tables together and attach bounded neighboring parent context."""
    if max_chars < 200 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Invalid chunk size or overlap")
    sections = [s.strip() for s in re.split(r"\n\s*\n", document.text) if s.strip()]
    if not sections:
        raise ValueError("Document must contain non-whitespace evidence")
    units: list[str] = []
    for section in sections:
        if len(section) <= max_chars:
            units.append(section)
        elif _is_table(section):
            units.extend(_split_table(section, max_chars))
        else:
            units.extend(_split_prose(section, max_chars, overlap))
    texts: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) + 2 > max_chars:
            texts.append(current)
            current = unit
        else:
            current = f"{current}\n\n{unit}" if current else unit
    if current:
        texts.append(current)
    chunks = []
    for ordinal, text in enumerate(texts):
        digest = hashlib.sha256(text.encode()).hexdigest()
        identifier = hashlib.sha256(
            f"{document.tenant_id}\0{document.id}\0{ordinal}\0{digest}".encode()
        ).hexdigest()
        parent = "\n\n".join(texts[max(0, ordinal - 1) : ordinal + 2])
        chunks.append(
            Chunk(
                id=identifier,
                document_id=document.id,
                tenant_id=document.tenant_id,
                title=document.title,
                text=text,
                parent_text=parent,
                source=document.source,
                published_at=document.published_at,
                allowed_groups=document.allowed_groups,
                ordinal=ordinal,
                content_hash=digest,
            )
        )
    return chunks
