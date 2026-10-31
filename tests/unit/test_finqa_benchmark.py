import json
import sys
from argparse import Namespace
from datetime import date
from pathlib import Path

import pytest
from pydantic import SecretStr

from ledger_rag.core.models import Chunk, Document, Hit
from ledger_rag.evaluation.benchmarks.finqa import (
    Case,
    answer_correct,
    bench_settings,
    main,
    numbers,
    page_document,
    page_metrics,
    summarize,
    table_markdown,
)
from ledger_rag.ingestion.chunking import chunk_document, contextual_text


def _doc(text: str) -> Document:
    return Document(
        id="d",
        tenant_id="t",
        title="ACME annual report 2020, page 4",
        text=text,
        source="https://example.test",
        published_at=date(2020, 12, 31),
    )


def test_oversized_table_splits_on_rows_and_repeats_header() -> None:
    header = "| item | 2020 | 2019 |\n|---|---|---|"
    rows = [f"| line item number {i} | {i * 100} | {i * 90} |" for i in range(60)]
    chunks = chunk_document(_doc(header + "\n" + "\n".join(rows)), max_chars=400, overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.startswith(header)
        assert all(line.startswith("|") and line.endswith("|") for line in chunk.text.splitlines())
    joined = "\n".join(chunk.text for chunk in chunks)
    assert all(row in joined for row in rows)


def test_oversized_prose_prefers_sentence_boundaries() -> None:
    sentences = [f"Sentence {i} reports revenue of {i} million dollars." for i in range(60)]
    chunks = chunk_document(_doc(" ".join(sentences)), max_chars=300, overlap=30)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.endswith(".")
        assert len(chunk.text) <= 300


def test_single_huge_sentence_falls_back_to_windows() -> None:
    chunks = chunk_document(_doc("x" * 1000), max_chars=300, overlap=50)
    assert all(len(c.text) <= 300 for c in chunks)
    assert "".join(c.text for c in chunks).count("x") >= 1000


def test_contextual_text_prefixes_title() -> None:
    chunk = chunk_document(_doc("Revenue rose."))[0]
    assert contextual_text(chunk) == "ACME annual report 2020, page 4\nRevenue rose."


EXAMPLE = {
    "filename": "ACME/2020/page_4.pdf",
    "pre_text": ["revenue grew strongly .", "."],
    "post_text": ["margins were stable ."],
    "table": [["", "2020", "2019"], ["net sales", "$ 120", "$ 100"]],
}


def test_page_document_and_units() -> None:
    document, units = page_document(EXAMPLE)
    assert document.id == "ACME/2020/page_4.pdf"
    assert document.allowed_groups == ["company:ACME"]
    assert "| net sales | $ 120 | $ 100 |" in document.text
    assert units["table_1"] == "| net sales | $ 120 | $ 100 |"
    assert units["text_2"] == "margins were stable ."
    markdown, rows = table_markdown([["a|b", "c"], ["1"]])
    assert rows == ["| a/b | c |", "| 1 |  |"]
    assert markdown.splitlines()[1] == "|---|---|"


def _hit(document_id: str, text: str) -> Hit:
    return Hit(
        chunk=Chunk(
            id=document_id + text,
            document_id=document_id,
            tenant_id="t",
            title="t",
            text=text,
            parent_text=text,
            source="s",
            published_at=date(2020, 1, 1),
            ordinal=0,
            content_hash="h",
        ),
        score=1.0,
    )


def test_page_metrics_rank_and_evidence() -> None:
    case = Case("q", "?", "gold", "ACME", ["row a", "row b"], 1.0, "")
    hits = [_hit("other", "row a"), _hit("other", "x"), _hit("gold", "row a and more")]
    metrics = page_metrics(hits, case)
    assert metrics["page_hit@1"] == 0.0
    assert metrics["page_hit@3"] == 1.0
    assert metrics["page_mrr@10"] == pytest.approx(0.5)
    assert metrics["evidence_recall@5"] == 0.5
    assert page_metrics([], case)["page_mrr@10"] == 0.0


@pytest.mark.parametrize(
    "text,gold,expected",
    [
        ("20.0% [C1]. Inputs were 120 and 100.", 0.2, True),
        ("The change was 14.14 percent [C1]", 14.1422, True),
        ("$1,234.5 million [C1]", 1234.5, True),
        ("-3.2% decline [C1]", -0.032, True),
        ("100 and then 20% [C1]", 0.2, False),
        ("yes, it increased [C1]", "yes", True),
        ("no [C1]", "yes", False),
        ("no numbers here", 3.0, False),
    ],
)
def test_answer_correct(text: str, gold: float | str, expected: bool) -> None:
    assert answer_correct(text, gold) is expected


def test_numbers_ignore_citation_markers() -> None:
    assert numbers("[C12] total 1,000 and 2.5") == [1000.0, 2.5]


def test_summarize_bootstrap_bounds() -> None:
    result = summarize([{"m": 1.0}, {"m": 0.0}, {"m": 1.0}, {"m": 1.0}])
    assert result["m"]["mean"] == 0.75
    low, high = result["m"]["ci95"]
    assert 0 <= low <= 0.75 <= high <= 1


@pytest.mark.parametrize(
    ("mode", "oversampling"), [("none", 2.0), ("scalar", 4.0), ("binary", 1.5)]
)
def test_benchmark_settings_match_quantized_collection(mode: str, oversampling: float) -> None:
    args = Namespace(
        qdrant_url="http://localhost:6333",
        collection=f"finqa_{mode}",
        embedding_url="http://localhost:8080/v1",
        embedding_key="test-key",
        embedding_model="test-model",
        embedding_space_id="test-space-v1",
        embedding_dimensions=768,
        query_prefix="query: ",
        rerank_candidates=20,
        quantization=mode,
        oversampling=oversampling,
    )
    settings = bench_settings(args)
    assert settings.qdrant_quantization == mode
    assert settings.qdrant_oversampling == oversampling
    assert settings.qdrant_collection == args.collection
    assert settings.embedding_api_key == SecretStr("test-key")
    assert settings.embedding_space_id == "test-space-v1"


@pytest.mark.parametrize("selected_split", [None, "train", "dev", "test"])
def test_cli_split_selects_questions_but_preserves_full_corpus(
    selected_split: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sizes = {"train": 1, "dev": 2, "test": 3}
    for page, (split, count) in enumerate(sizes.items(), 1):
        rows = [
            {
                **EXAMPLE,
                "filename": f"ACME/2020/page_{page}.pdf",
                "id": f"{split}-{i}",
                "qa": {"question": f"Question {split} {i}?", "exe_ans": 1, "gold_inds": {}},
            }
            for i in range(count)
        ]
        (tmp_path / f"{split}.json").write_text(json.dumps(rows))
    output = tmp_path / "report.json"
    argv = ["finqa", "stats", "--data", str(tmp_path), "--output", str(output), "--limit", "1"]
    if selected_split is not None:
        argv.extend(["--split", selected_split])
    monkeypatch.setattr(sys, "argv", argv)
    main()
    dataset = json.loads(output.read_text())["dataset"]
    effective_split = selected_split or "test"
    assert dataset["corpus_pages"] == 3
    assert dataset["question_split"] == effective_split
    assert dataset["split_questions"] == sizes[effective_split]
    assert dataset["selected_questions"] == 1
    if effective_split == "test":
        assert dataset["test_questions"] == 1
    else:
        assert "test_questions" not in dataset


def test_cli_rejects_negative_case_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["finqa", "stats", "--limit", "-1"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
