import hashlib
import re

from ledger_rag.models import Chunk, Document


def chunk_document(document: Document, max_chars: int = 1800, overlap: int = 200) -> list[Chunk]:
    """Keep paragraphs/tables together and attach bounded neighboring parent context."""
    if max_chars < 200 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Invalid chunk size or overlap")
    sections = [s.strip() for s in re.split(r"\n\s*\n", document.text) if s.strip()]
    units: list[str] = []
    for section in sections:
        if len(section) <= max_chars:
            units.append(section)
        else:
            step = max_chars - overlap
            units.extend(section[i : i + max_chars] for i in range(0, len(section), step))
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
