"""Structure-aware chunking for financial disclosures.

Paragraphs and tables are kept intact where possible. Oversized markdown tables are
split on row boundaries and every continuation repeats the header rows, so a
retrieved fragment never contains numbers without their column labels. Oversized
prose is split on sentence boundaries before falling back to overlapping windows.

Chunk profiles version the text that is embedded, sparse-indexed and reranked:

- ``v1``: ``"{title}\n{text}"``. Byte-identical to the original pipeline so existing
  collections stay valid; ``Chunk.index_text`` is left unset.
- ``v2``: a deterministic document header (filing, section, table caption) plus the
  chunk body in which every Markdown table is replaced by a row-wise linearization
  (``"<row label> | <column header>: <value>"``). Each table unit starts with a header
  block: the top-left cell as ``Units:``/``Rows:`` (dropped when the rendered caption
  already contains it) and ``Column:`` lines for headers no row line carries in full.
  ``text`` keeps the Markdown table for display and citation.
- ``v3``: ``v2`` plus a cached per-document LLM summary line in the header.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Literal, get_args

from ledger_rag.core.models import Chunk, Document
from ledger_rag.ingestion.context import (
    SectionHeadings,
    document_context,
    render_header,
    table_caption,
)

ChunkProfile = Literal["v1", "v2", "v3"]
CHUNK_PROFILES: tuple[ChunkProfile, ...] = get_args(ChunkProfile)
LABEL_CHARS = 160
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_SEPARATOR = re.compile(r"\s*\|[\s:|-]+\|\s*")
_CELL_SPLIT = re.compile(r"(?<!\\)\|")
_UNITS = re.compile(r"\b(?:millions?|thousands?|billions?|dollars|percent(?:age)?)\b|[$%€£¥]")


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
    if chunk.index_text is not None:
        return chunk.index_text
    return f"{chunk.title}\n{chunk.text}"


def chunk_document(
    document: Document,
    max_chars: int = 1800,
    overlap: int = 200,
    *,
    profile: ChunkProfile = "v1",
    summary: str | None = None,
    index_chars: int | None = None,
) -> list[Chunk]:
    """Keep paragraphs/tables together and attach bounded neighboring parent context.

    ``index_chars`` bounds the v2/v3 indexed body (default ``2 * max_chars``); the
    rendered header adds at most a few hundred characters on top of it.
    """
    if max_chars < 200 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Invalid chunk size or overlap")
    if profile not in CHUNK_PROFILES:
        raise ValueError(f"Unknown chunk profile {profile!r}")
    if profile == "v3" and not (summary and summary.strip()):
        raise ValueError("Chunk profile v3 requires a document summary")
    if profile != "v3" and summary is not None:
        raise ValueError("Summaries are only indexed by chunk profile v3")
    if profile != "v1":
        return _chunk_contextual(
            document, max_chars, overlap, profile, summary, index_chars or 2 * max_chars
        )
    if index_chars is not None:
        raise ValueError("index_chars applies only to contextual chunk profiles")
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


@dataclass(frozen=True)
class _Unit:
    text: str
    index: str
    section: int
    offset: int
    caption: str | None = None
    solo: bool = False
    corner: str = ""

    @property
    def index_size(self) -> int:
        # Upper bound: the corner line may be dropped when the rendered caption repeats it.
        return len(self.index) + (len(_corner_line(self.corner)) + 1 if self.corner else 0)


def _cells(line: str) -> list[str]:
    body = line.strip()
    body = body.removeprefix("|")
    if body.endswith("|") and not body.endswith("\\|"):
        body = body[:-1]
    return [cell.strip().replace("\\|", "|") for cell in _CELL_SPLIT.split(body)]


def _label(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= LABEL_CHARS else text[: LABEL_CHARS - 1].rstrip() + "…"


def _full(text: str) -> str:
    return " ".join(text.split())


def _corner_line(cell: str) -> str:
    """The top-left header cell names the units or the row dimension (``$ in millions``)."""
    kind = "Units" if _UNITS.search(cell.lower()) else "Rows"
    return f"{kind}: {_full(cell)}"


def table_header_lines(columns: list[str], rows: list[list[str]]) -> list[str]:
    """Column headers that the row lines would not carry in full.

    A header is emitted when it is clipped in row lines or no row has a value under it.
    The top-left cell is handled separately (see ``_corner_line``).
    """
    lines = []
    for position, header in enumerate(columns[1:], 1):
        if not header.strip():
            continue
        has_value = any(position < len(cells) and cells[position] for cells in rows)
        if not has_value or _label(header) != _full(header):
            lines.append(f"Column: {_full(header)}")
    return lines


def linearize_row(columns: list[str] | None, cells: list[str]) -> list[str]:
    """One line per value: ``row label | column header: value``; never splits a value.

    Labels are clipped to ``LABEL_CHARS``; a clipped row label is emitted once in full.
    """
    if not cells:
        return []
    label = _label(cells[0])
    lines: list[str] = [_full(cells[0])] if label != _full(cells[0]) else []
    for position, value in enumerate(cells[1:], 1):
        if not value:
            continue
        header = _label(columns[position]) if columns and position < len(columns) else ""
        cell = f"{header}: {value}" if header else value
        lines.append(f"{label} | {cell}" if label else cell)
    return lines or ([label] if label else [])


def _table_preamble(columns: list[str], rows: list[list[str]]) -> tuple[str, list[str]]:
    return columns[0] if columns else "", table_header_lines(columns, rows)


def linearize_table(markdown: str) -> str:
    """Row-wise linearization of a pipe-Markdown table (header row = column labels)."""
    lines = [line for line in markdown.splitlines() if line.strip()]
    if not lines:
        return ""
    header_count = _header_count(lines)
    rows = lines[header_count:]
    if not rows:
        return "\n".join(linearize_row(None, _cells(lines[0])))
    columns = _cells(lines[0])
    cells = [_cells(row) for row in rows]
    corner, header_lines = _table_preamble(columns, cells)
    body = [line for row in cells for line in linearize_row(columns, row)]
    return "\n".join([*([_corner_line(corner)] if corner else []), *header_lines, *body])


def _header_count(lines: list[str]) -> int:
    return 2 if len(lines) > 1 and _SEPARATOR.fullmatch(lines[1]) else 1


def _blocks(section: str) -> list[tuple[bool, str]]:
    """Split a section into prose and pipe-table runs, so caption + table sections work."""
    runs: list[tuple[bool, list[str]]] = []
    for line in section.splitlines():
        is_table = line.lstrip().startswith("|")
        if runs and runs[-1][0] == is_table:
            runs[-1][1].append(line)
        else:
            runs.append((is_table, [line]))
    blocks: list[tuple[bool, str]] = []
    for is_table, lines in runs:
        table = is_table and len(lines) >= 2
        text = "\n".join(lines).strip()
        if not text:
            continue
        if blocks and not table and not blocks[-1][0]:
            blocks[-1] = (False, blocks[-1][1] + "\n" + text)
        else:
            blocks.append((table, text))
    return blocks


def _pack_lines(lines: list[str], limit: int) -> list[str]:
    groups: list[list[str]] = []
    size = 0
    for line in lines:
        if groups and groups[-1] and size + len(line) + 1 <= limit:
            groups[-1].append(line)
            size += len(line) + 1
        else:
            groups.append([line])
            size = len(line)
    return ["\n".join(group) for group in groups]


def _table_units(
    block: str, section: int, offset: int, caption: str | None, max_chars: int, index_chars: int
) -> list[_Unit]:
    lines = block.splitlines()
    header_count = _header_count(lines)
    header = "\n".join(lines[:header_count])
    rows = lines[header_count:]
    if len(header) > max_chars // 2:
        # Unparseable shape: keep v1 windows and index them verbatim.
        return [
            _Unit(piece, piece, section, offset, caption) for piece in _windows(block, max_chars, 0)
        ]
    if not rows:
        # A header-only table (FinQA single-row tables) is linearized as one data row.
        parts = _pack_lines(linearize_row(None, _cells(lines[0])), index_chars) or [block]
        return [_Unit(block, part, section, offset, caption, solo=len(parts) > 1) for part in parts]
    columns = _cells(lines[0])
    corner, preamble = _table_preamble(columns, [_cells(row) for row in rows])
    # Every unit of the table repeats the header block, like the Markdown header rows.
    reserved = sum(len(line) + 1 for line in preamble)
    if corner:
        reserved += len(_corner_line(corner)) + 1
    units: list[_Unit] = []
    group: list[str] = []
    linear: list[str] = []

    def unit(text: str, body: list[str], solo: bool = False) -> _Unit:
        index = "\n".join([*preamble, *body])
        return _Unit(text, index, section, offset, caption, solo=solo, corner=corner)

    def flush() -> None:
        nonlocal group, linear
        if group:
            units.append(unit("\n".join([header, *group]), linear))
        group, linear = [], []

    for row in rows:
        row_lines = linearize_row(columns, _cells(row))
        row_index = sum(len(line) + 1 for line in row_lines) + reserved
        if len(header) + len(row) + 1 > max_chars:
            flush()
            units.extend(
                _Unit(f"{header}\n{piece}", f"{header}\n{piece}", section, offset, caption)
                for piece in _windows(row, max_chars - len(header) - 1, 0)
            )
            continue
        if row_index > index_chars:
            flush()
            text = f"{header}\n{row}"
            units.extend(
                unit(text, [part], solo=True)
                for part in _pack_lines(row_lines, index_chars - reserved)
            )
            continue
        text_size = len(header) + sum(len(r) + 1 for r in group) + len(row) + 1
        index_size = sum(len(line) + 1 for line in linear) + row_index
        if group and (text_size > max_chars or index_size > index_chars):
            flush()
        group.append(row)
        linear.extend(row_lines)
    flush()
    return units


def _contextual_units(
    sections: list[str], max_chars: int, overlap: int, index_chars: int
) -> list[_Unit]:
    units: list[_Unit] = []
    last_prose = ""
    for number, section in enumerate(sections):
        cursor = 0
        for is_table, block in _blocks(section):
            offset = max(section.find(block[:80], cursor), cursor)
            cursor = offset
            if is_table:
                caption = table_caption(last_prose) if last_prose else None
                units.extend(_table_units(block, number, offset, caption, max_chars, index_chars))
                last_prose = ""
                continue
            last_prose = block
            pieces = [block] if len(block) <= max_chars else _split_prose(block, max_chars, overlap)
            for piece in pieces:
                found = section.find(piece[:80], cursor)
                position = found if found >= 0 else cursor
                cursor = position
                units.append(_Unit(piece, piece, number, position))
    return units


def _unit_index(unit: _Unit, caption: str | None) -> str:
    if not unit.corner or (caption and _full(unit.corner).lower() in caption.lower()):
        return unit.index
    return f"{_corner_line(unit.corner)}\n{unit.index}" if unit.index else _corner_line(unit.corner)


def _chunk_contextual(
    document: Document,
    max_chars: int,
    overlap: int,
    profile: ChunkProfile,
    summary: str | None,
    index_chars: int,
) -> list[Chunk]:
    if index_chars < max_chars:
        raise ValueError("index_chars must be at least max_chars")
    sections = [s.strip() for s in re.split(r"\n\s*\n", document.text) if s.strip()]
    if not sections:
        raise ValueError("Document must contain non-whitespace evidence")
    units = _contextual_units(sections, max_chars, overlap, index_chars)
    groups: list[list[_Unit]] = []
    for unit in units:
        current = groups[-1] if groups else []
        text_size = sum(len(u.text) + 2 for u in current) + len(unit.text)
        index_size = sum(u.index_size + 2 for u in current) + unit.index_size
        if (
            not current
            or unit.solo
            or current[-1].solo
            or text_size > max_chars
            or index_size > index_chars
        ):
            groups.append([unit])
        else:
            current.append(unit)
    context = document_context(document.id, document.title)
    texts = ["\n\n".join(u.text for u in group) for group in groups]
    # The heading for a chunk is the last one in the sections before it, the part of its
    # own section before it, and the chunk; one indexed scan keeps this linear.
    headings = SectionHeadings("\n\n".join(sections))
    section_starts = [0]
    for section in sections:
        section_starts.append(section_starts[-1] + len(section) + 2)
    chunks = []
    for ordinal, group in enumerate(groups):
        first = group[0]
        text = texts[ordinal]
        heading = headings.heading(section_starts[first.section] + first.offset, text)
        caption = next((u.caption for u in group if u.caption), None)
        header = render_header(context, section=heading, caption=caption, summary=summary)
        index_text = header + "\n\n" + "\n\n".join(_unit_index(u, caption) for u in group)
        digest = hashlib.sha256(text.encode()).hexdigest()
        index_digest = hashlib.sha256(index_text.encode()).hexdigest()
        identifier = hashlib.sha256(
            f"{document.tenant_id}\0{document.id}\0{ordinal}\0{digest}\0{profile}\0{index_digest}".encode()
        ).hexdigest()
        chunks.append(
            Chunk(
                id=identifier,
                document_id=document.id,
                tenant_id=document.tenant_id,
                title=document.title,
                text=text,
                parent_text="\n\n".join(texts[max(0, ordinal - 1) : ordinal + 2]),
                source=document.source,
                published_at=document.published_at,
                allowed_groups=document.allowed_groups,
                ordinal=ordinal,
                content_hash=digest,
                index_text=index_text,
            )
        )
    return chunks
