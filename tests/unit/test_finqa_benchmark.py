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
    Corpus,
    bench_settings,
    corpus_sha256,
    gold_label_audit,
    holdout_views,
    load,
    main,
    page_document,
    page_metrics,
    require_valid_gold,
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


def test_page_metrics_rank_and_gold_unit_evidence() -> None:
    document, units = page_document(EXAMPLE)
    corpus = Corpus(documents={document.id: document}, units={document.id: units})
    evidence = corpus.evidence(document.id)
    gold = evidence.chunks[0]
    case = Case("q", "?", document.id, "ACME", [units["table_1"]], 1.0, "", gold_keys=["table_1"])
    other = _hit("other", "| net sales | $ 120 | $ 100 |")  # same text, wrong chunk identity
    hits = [other, _hit("x", "x"), Hit(chunk=gold, score=1.0)]
    metrics = page_metrics(hits, case, evidence)
    assert metrics["page_recall@1"] == 0.0
    assert metrics["page_recall@3"] == 1.0
    assert metrics["page_mrr@10"] == pytest.approx(1 / 3)
    # Substring matches in a different chunk earn no credit; identity does.
    assert metrics["evidence_recall@1"] == 0.0
    assert metrics["evidence_recall@3"] == 1.0
    assert metrics["evidence_precision@3"] == pytest.approx(1 / 3)
    assert metrics["evidence_mrr@10"] == pytest.approx(1 / 3)
    assert page_metrics([], case, evidence)["page_mrr@10"] == 0.0
    no_gold = Case("q", "?", document.id, "ACME", [], 1.0, "")
    assert page_metrics(hits, no_gold, evidence)["evidence_recall@5"] is None


def test_summarize_clusters_by_report() -> None:
    result = summarize([{"m": 1.0}, {"m": 0.0}, {"m": 1.0}, {"m": None}], ["a", "a", "b", "b"])
    assert result["m"]["mean"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["m"]["clusters"] == 2
    assert result["m"]["ineligible"] == 1
    low, high = result["m"]["ci95"]
    assert 0 <= low <= result["m"]["mean"] <= high <= 1


def _write_splits(tmp_path: Path, gold_inds: dict[str, str] | None) -> None:
    for page, split in enumerate(("train", "dev", "test"), 1):
        qa: dict[str, object] = {"question": f"{split}?", "exe_ans": 1.0, "answer": "1"}
        if gold_inds is not None:
            qa["gold_inds"] = gold_inds
        company = "ACME" if split != "dev" else "NEWCO"
        row = {**EXAMPLE, "filename": f"{company}/2020/page_{page}.pdf", "id": split, "qa": qa}
        (tmp_path / f"{split}.json").write_text(json.dumps([row]))


def test_unknown_and_missing_gold_keys_are_counted_and_fail_loudly(tmp_path: Path) -> None:
    _write_splits(tmp_path, {"text_0": "x", "table_9": "missing row"})
    _, cases = load(tmp_path, "dev")
    assert cases[0].gold_keys == ["text_0"]
    assert cases[0].unknown_gold_keys == ["table_9"]
    audit = gold_label_audit(cases)
    assert audit["unknown_gold_keys"] == 1
    assert audit["cases_with_unknown_gold_keys"] == ["dev"]
    with pytest.raises(ValueError, match="unknown keys"):
        require_valid_gold(cases)
    _write_splits(tmp_path, None)
    _, cases = load(tmp_path, "dev")
    assert gold_label_audit(cases)["cases_missing_gold_inds"] == 1
    with pytest.raises(ValueError, match="missing gold_inds"):
        require_valid_gold(cases)


def test_holdout_views_are_company_and_report_disjoint(tmp_path: Path) -> None:
    _write_splits(tmp_path, {"text_0": "x"})
    corpus, cases = load(tmp_path, "dev")
    views = holdout_views(corpus, cases)
    assert views["company_unseen_in_train"] == {"dev"}
    assert views["report_unseen_in_train"] == {"dev"}
    assert views["company_fold_holdout"] | views["company_fold_selection"] == {"dev"}
    assert cases[0].report == "NEWCO/2020"


def test_rebuilt_corpus_fingerprint_matches_bulk_ingestion_definition(tmp_path: Path) -> None:
    import hashlib

    _write_splits(tmp_path, {"text_0": "x"})
    corpus, _ = load(tmp_path, "dev")
    digest = hashlib.sha256()
    for document in corpus.documents.values():
        digest.update(document.model_dump_json().encode() + b"\n")
    assert corpus_sha256(corpus) == digest.hexdigest()


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
    if selected_split == "test":
        argv.append("--unseal-test")
    monkeypatch.setattr(sys, "argv", argv)
    main()
    dataset = json.loads(output.read_text())["dataset"]
    effective_split = selected_split or "dev"
    assert dataset["corpus_pages"] == 3
    assert dataset["question_split"] == effective_split
    assert dataset["split_questions"] == sizes[effective_split]
    assert dataset["selected_questions"] == 1
    assert dataset["test_split_unsealed"] is (effective_split == "test")
    if effective_split == "test":
        assert dataset["test_questions"] == 1
    else:
        assert "test_questions" not in dataset


def test_cli_rejects_negative_case_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["finqa", "stats", "--limit", "-1"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_cli_test_split_is_sealed_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["finqa", "stats", "--split", "test"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "extra",
    [["--configs", "hybrid:bogus"], ["--answer-modes", "retrieved@0"], ["--llm-concurrency", "9"]],
)
def test_cli_rejects_invalid_arms_modes_and_concurrency(
    extra: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["finqa", "stats", *extra])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
