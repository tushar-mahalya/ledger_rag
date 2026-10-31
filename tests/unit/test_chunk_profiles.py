import hashlib
import json
from datetime import date

import pytest

from ledger_rag.core.models import Chunk, Document
from ledger_rag.ingestion.chunking import (
    chunk_document,
    contextual_text,
    linearize_row,
    linearize_table,
)
from ledger_rag.ingestion.context import (
    document_context,
    document_hash,
    render_header,
    section_heading,
    table_caption,
)

ROWS = "\n".join(f"| revenue line {i} | $ {i}00 | $ {i}50 |" for i in range(60))
TEXT = (
    "item 7 . management 2019s discussion and analysis. " * 3
    + "the following table summarizes revenue ( in millions ) :\n\n"
    + "| | 2009 | 2008 |\n|---|---|---|\n"
    + ROWS
    + "\n\n"
    + "Prose sentence about liquidity. " * 120
)
# Digest of the pre-profile chunker's output for this document (git d65bf0b).
V1_GOLDEN = "4f143cb21c853e53b60413a7ef958e4e3c9bc1c1c59f50611e0ef6cc3ff1fc9c"


@pytest.fixture
def filing() -> Document:
    return Document(
        id="ADI/2009/page_49.pdf",
        tenant_id="finqa",
        title="ADI annual report 2009, page 49",
        text=TEXT,
        source="s",
        published_at=date(2009, 12, 31),
    )


def _fingerprint(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for c in chunks:
        row = [c.id, c.text, c.parent_text, c.ordinal, c.content_hash, contextual_text(c)]
        digest.update(json.dumps(row).encode())
    return digest.hexdigest()


def test_v1_profile_is_byte_identical_to_the_original_chunker(filing: Document) -> None:
    chunks = chunk_document(filing)
    assert _fingerprint(chunks) == V1_GOLDEN
    assert chunk_document(filing, profile="v1") == chunks
    assert all(c.index_text is None for c in chunks)
    assert all(contextual_text(c) == f"{c.title}\n{c.text}" for c in chunks)


def test_v1_payload_without_index_text_round_trips_as_v1(filing: Document) -> None:
    chunk = chunk_document(filing)[0]
    legacy = chunk.model_dump(mode="json")
    del legacy["index_text"]
    assert contextual_text(Chunk.model_validate(legacy)) == f"{chunk.title}\n{chunk.text}"


def test_v2_keeps_markdown_text_and_indexes_header_and_linearized_rows(filing: Document) -> None:
    chunks = chunk_document(filing, profile="v2")
    assert chunks == chunk_document(filing, profile="v2")
    assert len({c.id for c in chunks}) == len(chunks)
    assert not {c.id for c in chunks} & {c.id for c in chunk_document(filing)}
    table_chunks = [c for c in chunks if "| revenue line 3 |" in c.text]
    assert table_chunks
    table = table_chunks[0]
    assert "| | 2009 | 2008 |\n|---|---|---|" in table.text
    assert table.index_text is not None
    header, _, body = table.index_text.partition("\n\n")
    assert header.splitlines() == [
        "Filing: ADI Form 10-K, fiscal year 2009, page 49",
        "Section: Item 7. Management's Discussion and Analysis",
        "Table: the following table summarizes revenue ( in millions ) :",
    ]
    assert "revenue line 3 | 2009: $ 300\nrevenue line 3 | 2008: $ 350" in body
    assert "|---|" not in body
    for chunk in chunks:
        assert len(chunk.text) <= 1800
        assert chunk.index_text is not None
        assert len(chunk.index_text.partition("\n\n")[2]) <= 3600
        assert contextual_text(chunk) == chunk.index_text
        for line in chunk.index_text.splitlines():
            if line.startswith("revenue line"):
                assert ": $ " in line  # a value always stays with its column header


def test_v2_continuations_repeat_header_and_caption(filing: Document) -> None:
    chunks = chunk_document(filing, max_chars=400, profile="v2")
    tables = [c for c in chunks if "revenue line" in c.text]
    assert len(tables) > 2
    for chunk in tables:
        table_header = "| | 2009 | 2008 |\n|---|---|---|"
        assert chunk.text.startswith(table_header) or "following" in chunk.text
        assert chunk.index_text is not None and "\nTable: the following table" in chunk.index_text
        assert len(chunk.text) <= 400
        assert len(chunk.index_text.partition("\n\n")[2]) <= 800


def test_v2_wide_row_linearization_is_split_on_value_boundaries() -> None:
    columns = "| label | " + " | ".join(f"fy{i}" for i in range(20)) + " |"
    separator = "|" + "---|" * 21
    row = (
        "| net income attributable to "
        + "common " * 20
        + "| "
        + " | ".join(f"{i}.5" for i in range(20))
    )
    document = Document(
        id="doc",
        tenant_id="t",
        title="Wide",
        text=f"{columns}\n{separator}\n{row}",
        source="s",
        published_at=date(2020, 1, 1),
    )
    chunks = chunk_document(document, max_chars=1800, profile="v2", index_chars=1800)
    assert len(chunks) > 1
    assert len({c.text for c in chunks}) == 1
    lines = [line for c in chunks for line in (c.index_text or "").split("\n\n", 1)[1].splitlines()]
    assert lines == linearize_table(document.text).splitlines()
    assert all(len((c.index_text or "").split("\n\n", 1)[1]) <= 1800 for c in chunks)


def test_caption_plus_table_section_is_recognized_as_a_table() -> None:
    document = Document(
        id="doc",
        tenant_id="t",
        title="Caption",
        text="Revenue by segment (in millions):\n| segment | 2020 |\n|---|---|\n| cloud | 5 |",
        source="s",
        published_at=date(2020, 1, 1),
    )
    [chunk] = chunk_document(document, profile="v2")
    assert chunk.text == document.text.replace(":\n|", ":\n\n|")
    assert chunk.index_text == (
        "Document: Caption\nTable: Revenue by segment (in millions):\n\n"
        "Revenue by segment (in millions):\n\ncloud | 2020: 5"
    )


def test_linearization_handles_header_only_and_empty_cells() -> None:
    assert linearize_table("| employee separations | liability $ 20963 | |\n|---|---|---|") == (
        "employee separations | liability $ 20963"
    )
    assert linearize_row(["", "2009"], ["", "5"]) == ["2009: 5"]
    assert linearize_row(["", "2009"], ["operating activities", ""]) == ["operating activities"]
    assert linearize_table("| k | a \\| b |\n|---|---|\n| r | 1 |") == "r | a | b: 1"


def test_document_context_never_invents_company_names() -> None:
    context = document_context("HWM/2018/page_12.pdf", "HWM annual report 2018, page 12")
    assert (context.ticker, context.fiscal_year, context.page, context.form_type) == (
        "HWM",
        2018,
        12,
        "10-K",
    )
    assert render_header(context) == "Filing: HWM Form 10-K, fiscal year 2018, page 12"
    other = document_context("upload-1", "Q3 board memo")
    assert render_header(other, section="Risk") == "Document: Q3 board memo\nSection: Risk"


def test_section_heading_and_caption_detection() -> None:
    assert section_heading("abc notes to consolidated financial statements x") == (
        "Notes to Consolidated Financial Statements"
    )
    assert section_heading("item 7a . quantitative and qualitative disclosures") == (
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk"
    )
    assert section_heading("plain prose about revenue") is None
    assert table_caption("Some prose. The company had growth.") is None
    assert table_caption("Intro. the following table presents debt :") == (
        "the following table presents debt :"
    )
    long = table_caption("x. the following table " + "word " * 100)
    assert long is not None and len(long) <= 240 and long.endswith("…")


def test_profile_argument_validation(filing: Document) -> None:
    with pytest.raises(ValueError, match="Unknown chunk profile"):
        chunk_document(filing, profile="v9")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires a document summary"):
        chunk_document(filing, profile="v3")
    with pytest.raises(ValueError, match="only indexed by chunk profile v3"):
        chunk_document(filing, profile="v2", summary="x")
    with pytest.raises(ValueError, match="index_chars"):
        chunk_document(filing, index_chars=4000)
    with pytest.raises(ValueError, match="index_chars"):
        chunk_document(filing, profile="v2", index_chars=100)


def test_v3_adds_a_bounded_summary_line(filing: Document) -> None:
    chunks = chunk_document(filing, profile="v3", summary="ADI discusses revenue. " * 60)
    v2 = chunk_document(filing, profile="v2")
    assert [c.text for c in chunks] == [c.text for c in v2]
    for chunk in chunks:
        header = (chunk.index_text or "").partition("\n\n")[0]
        summary = [line for line in header.splitlines() if line.startswith("Summary: ")]
        assert len(summary) == 1 and len(summary[0]) <= 610
    assert document_hash(filing.title, filing.text) == document_hash(filing.title, filing.text)
    assert document_hash("a", "b") != document_hash("a", "c")
