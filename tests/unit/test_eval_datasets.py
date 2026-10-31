import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ledger_rag.evaluation.benchmarks.finqa import page_document
from ledger_rag.evaluation.datasets import build, convfinqa, financebench, tatqa
from ledger_rag.evaluation.datasets.schema import (
    CorpusDocument,
    QAItem,
    Unit,
    read_jsonl,
    render_table,
    write_jsonl,
)
from ledger_rag.ingestion.bulk import jsonl_documents
from ledger_rag.ingestion.chunking import chunk_document


def finqa_page(**extra: Any) -> dict[str, Any]:
    return {
        "filename": "ACME/2019/page_7.pdf",
        "pre_text": ["revenue grew in fiscal 2019 .", ""],
        "post_text": ["margins were stable ."],
        "table": [["", "2019", "2018"], ["revenue", "$ 120", "$ 100"], ["cost", "80", "70"]],
        **extra,
    }


def single_conversation() -> dict[str, Any]:
    return finqa_page(
        id="Single_ACME/2019/page_7.pdf-1",
        qa={"question": "what was the growth?", "gold_inds": {"table_1": "x", "text_0": "y"}},
        annotation={
            "dialogue_break": ["what was revenue in 2019?", "and in 2018?", "what is the growth?"],
            "turn_program": ["120", "100", "subtract(120, 100), divide(#0, 100)"],
            "exe_ans_list": [120.0, 100.0, 0.2],
            "qa_split": [0, 0, 0],
        },
    )


def double_conversation() -> dict[str, Any]:
    return finqa_page(
        id="Double_ACME/2019/page_7.pdf",
        qa_0={"question": "revenue change?", "gold_inds": {"table_1": "x"}},
        qa_1={"question": "did cost rise?", "gold_inds": {"table_2": "z", "text_99": "gone"}},
        annotation={
            "dialogue_break": ["what was the revenue change?", "did cost rise?"],
            "turn_program": ["subtract(120, 100)", "greater(80, 70)"],
            "exe_ans_list": [20.0, "yes"],
            "qa_split": [0, 1],
        },
    )


def test_units_validate_table_structure() -> None:
    with pytest.raises(ValidationError):
        Unit(unit_id="d#table", kind="table", text="|a|")
    with pytest.raises(ValidationError):
        Unit(unit_id="d#text", kind="text", text="a", rows=[["a"]])
    with pytest.raises(ValidationError):
        Unit(unit_id="d#t", kind="table", text="|a|", rows=[["a"]], header_rows=2)


def test_render_table_merges_multirow_headers_for_chunk_continuations() -> None:
    rows = [["", "Year ended", ""], ["", "2019", "2018"]] + [
        [f"line item {i}", str(i), str(i + 1)] for i in range(200)
    ]
    text = render_table(rows, header_rows=2)
    assert text.splitlines()[:2] == ["|  | Year ended 2019 | 2018 |", "|---|---|---|"]
    document = CorpusDocument(
        id="t",
        tenant_id="tatqa",
        title="Financial report excerpt",
        text=text,
        source="urn:test",
        published_at=date(2019, 12, 31),
        dataset="tatqa",
    )
    chunks = chunk_document(document.to_document())
    assert len(chunks) > 1
    assert all(chunk.text.startswith("|  | Year ended 2019 | 2018 |") for chunk in chunks)


def test_convfinqa_maps_turns_history_programs_and_gold_units() -> None:
    items = list(convfinqa.qa_items(single_conversation(), "dev"))
    assert [item.id for item in items] == [
        f"convfinqa/dev/Single_ACME/2019/page_7.pdf-1/turn_{i}" for i in range(3)
    ]
    last = items[2]
    assert [(t.role, t.content) for t in last.turns] == [
        ("user", "what was revenue in 2019?"),
        ("assistant", "120"),
        ("user", "and in 2018?"),
        ("assistant", "100"),
    ]
    assert items[0].turns == []
    assert last.answer == "0.2" and last.answer_type == "number"
    assert last.program == "subtract(120, 100), divide(#0, 100)"
    assert last.gold_document_ids == ["ACME/2019/page_7.pdf"]
    assert last.gold_unit_ids == ["ACME/2019/page_7.pdf#table_1", "ACME/2019/page_7.pdf#text_0"]
    assert (last.license, last.usage) == ("MIT", "train_ok")
    assert last.metadata["finqa_question"] == "what was the growth?"


def test_convfinqa_double_conversations_use_each_turns_source_question() -> None:
    first, second = convfinqa.qa_items(double_conversation(), "train")
    assert first.gold_unit_ids == ["ACME/2019/page_7.pdf#table_1"]
    # Unknown upstream unit keys are dropped instead of inventing evidence.
    assert second.gold_unit_ids == ["ACME/2019/page_7.pdf#table_2"]
    assert (second.answer, second.answer_type) == ("yes", "yes_no")
    assert second.metadata["conversation_type"] == "double"
    assert second.metadata["qa_split"] == 1


def test_convfinqa_rejects_misaligned_turn_lists() -> None:
    example = single_conversation()
    example["annotation"]["qa_split"] = [0]
    with pytest.raises(ValueError, match="misaligned"):
        list(convfinqa.qa_items(example, "dev"))


def test_convfinqa_documents_are_finqa_documents_with_structured_units() -> None:
    document = convfinqa.corpus_document(single_conversation())
    assert document.to_document() == page_document(single_conversation())[0]
    assert (document.company, document.fiscal_year, document.tenant_id) == ("ACME", 2019, "finqa")
    units = {unit.unit_id: unit for unit in document.units}
    assert "ACME/2019/page_7.pdf#text_1" not in units
    row = units["ACME/2019/page_7.pdf#table_1"]
    assert row.rows == [["", "2019", "2018"], ["revenue", "$ 120", "$ 100"]]
    assert row.header_rows == 1


def test_convfinqa_coverage_counts_ids_and_content() -> None:
    page = single_conversation()
    other = finqa_page(filename="ZETA/2010/page_1.pdf", id="Single_ZETA/2010/page_1.pdf-1")
    changed = finqa_page(filename="BETA/2011/page_2.pdf", id="Single_BETA/2011/page_2.pdf-1")
    finqa_docs = {
        page["filename"]: page_document(page)[0],
        "BETA/2011/page_2.pdf": page_document({**changed, "post_text": ["different"]})[0],
    }
    report = convfinqa.finqa_coverage({"dev": [page, changed], "test": [other]}, finqa_docs)
    assert report["unique_pages"] == 3
    assert report["pages_in_finqa"] == 2
    assert report["pages_identical_to_finqa"] == 1
    assert report["missing_page_ids"] == ["ZETA/2010/page_1.pdf"]
    assert report["mismatched_page_ids"] == ["BETA/2011/page_2.pdf"]
    assert report["id_coverage"] == pytest.approx(2 / 3)


def tatqa_context() -> dict[str, Any]:
    return {
        "table": {
            "uid": "table-1",
            "table": [
                ["", "Year Ended", ""],
                ["(In millions)", "2019", "2018"],
                ["Current:", "", ""],
                ["Revenue", "$1,200", "$1,000"],
                ["Margin", "12%", "10%"],
            ],
        },
        "paragraphs": [
            {"uid": "p2", "order": 2, "text": "Margin improved on pricing."},
            {"uid": "p1", "order": 1, "text": "Revenue grew strongly."},
        ],
        "questions": [
            {
                "uid": "q-arith",
                "order": 2,
                "question": "What was the change in margin?",
                "answer": 2,
                "derivation": "12-10",
                "answer_type": "arithmetic",
                "answer_from": "table",
                "rel_paragraphs": [],
                "req_comparison": False,
                "scale": "percent",
            },
            {
                "uid": "q-spans",
                "order": 1,
                "question": "What were revenues?",
                "answer": ["$1,200", "$1,000"],
                "derivation": "",
                "answer_type": "multi-span",
                "answer_from": "table-text",
                "rel_paragraphs": ["1", "7"],
                "req_comparison": False,
                "scale": "million",
            },
        ],
    }


def test_tatqa_document_keeps_table_structure_and_excludes_answers() -> None:
    document = tatqa.corpus_document(tatqa_context(), "dev")
    assert document.id == "tatqa/table-1"
    assert (document.fiscal_year, document.published_at) == (2019, date(2019, 12, 31))
    assert document.company is None and document.allowed_groups == []
    table = document.units[0]
    assert table.kind == "table" and table.header_rows == 2
    assert table.rows[2] == ["Current:", "", ""]
    assert [unit.unit_id for unit in document.units[1:]] == [
        "tatqa/table-1#paragraph_1",
        "tatqa/table-1#paragraph_2",
    ]
    assert document.text.splitlines()[0] == "| (In millions) | Year Ended 2019 | 2018 |"
    assert "12-10" not in document.text
    assert (
        "dataset_raw/tatqa_dataset_test_gold.json"
        in tatqa.corpus_document(tatqa_context(), "test").source
    )


def test_tatqa_undated_context_never_hidden_by_date_filters() -> None:
    context = tatqa_context()
    context["table"]["table"] = [["", "Current", "Prior"], ["Revenue", "5", "4"]]
    document = tatqa.corpus_document(context, "train")
    assert document.fiscal_year is None
    assert document.published_at == tatqa.UNKNOWN_DATE


def test_tatqa_questions_map_types_scales_and_evidence() -> None:
    context = tatqa_context()
    document = tatqa.corpus_document(context, "dev")
    spans, arithmetic = tatqa.qa_items(context, document, "dev")
    assert spans.id == "tatqa/dev/q-spans"
    assert (spans.answer, spans.answers, spans.answer_type) == (
        "$1,200 | $1,000",
        ["$1,200", "$1,000"],
        "multi_span",
    )
    assert spans.gold_unit_ids == ["tatqa/table-1#table", "tatqa/table-1#paragraph_1"]
    assert spans.program is None and spans.scale == "million"
    assert (arithmetic.answer, arithmetic.answer_type, arithmetic.program) == (
        "2",
        "percent",
        "12-10",
    )
    assert (arithmetic.license, arithmetic.usage) == ("CC-BY-4.0", "train_ok")


@pytest.mark.parametrize(
    "cell",
    [
        "$29,146 *",
        "$22,185*",
        "3,646,581 (i)",
        "74,100 (1)",
        "3,460.0p",
        "(0.2)%",
        "$(99)",
        "(51)",
        "–138",
        "64% - 69%",
        "6.1 - 6.3",
        "-",
        "—",
        "–",
        "- -",
        "$—",
        "n/a",
        "nil",
        "None",
        "11th June",
    ],
)
def test_tatqa_body_values(cell: str) -> None:
    assert tatqa.is_body_value(cell)


@pytest.mark.parametrize(
    "cell",
    [
        "2018",
        "FY2019",
        "2019 vs 2018",
        "2 0 1 8",
        "2018 1",
        "2021-2022",
        "1 - 3",
        "Year ended December 31, 2019",
        "December 31",
        "Q1",
        "%",
        "%Chg",
        "Level 1",
        "Total Number of Shares Purchased (1)",
        "($ in millions)",
    ],
)
def test_tatqa_header_tokens(cell: str) -> None:
    assert not tatqa.is_body_value(cell)


def test_tatqa_footnoted_values_end_the_header() -> None:
    # tatqa/d37c6159-a2af-4d21-88bc-57491ef0d175 (TAT-QA train)
    rows = [
        ["($ in millions)", "", "", "", ""],
        ["For the year ended December 31:", "2018", "2017", "Yr.-to-Yr. Percent Change", "Adj."],
        [
            "Global Technology Services external revenue",
            "$29,146 *",
            "$29,213 *",
            "(0.2)%",
            "(0.8)%",
        ],
        ["Infrastructure & Cloud Services", "$22,185*", "$22,016*", "0.8%", "0.0%"],
        ["Technology Support Services", "6,961", "7,196", "(3.3)", "(3.5)"],
    ]
    assert tatqa.header_row_count(rows) == 2
    assert tatqa.fiscal_year(rows, 2) == 2018
    rendered = render_table(rows, 2)
    assert "| Global Technology Services external revenue | $29,146 * |" in rendered


def test_tatqa_dash_and_footnoted_rows_end_the_header() -> None:
    # tatqa/7fdfb900-e531-45a1-836e-b7b08f4e7336 (TAT-QA train), headers shortened
    rows = [
        ["Type of Payment", "Termination ($)", "Death ($)", "Change In Control ($)", "Both ($)"],
        ["Cash Compensation", "-", "-", "-", "-"],
        ["Accelerated Vesting of Stock Option Awards", "-", "-", "-", "74,100 (1)"],
        ["Accelerated Vesting of Restricted Stock Unit Awards", "-", "-", "-", "-"],
        ["Accelerated Vesting of Performance Awards", "- -", "204,000 (2)", "-", "204,000 (2)"],
        ["Medical and Other Benefits", "-", "-", "-", "-"],
        ["Total", "-", "204,000", "-", "278,100"],
    ]
    assert tatqa.header_row_count(rows) == 1


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([["", "2019", "2018"], ["Revenue", "3,460.0p", "2,797.0p"], ["Cost", "1", "2"]], 1),
        ([["", "2015 Grant"], ["Grant date", "11th June"], ["Price", "3,460.0p"]], 1),
        (
            [
                ["", "December 31", ""],
                ["", "2019", "2 0 1 8"],
                ["", "$ in thousands", ""],
                ["A", "1", "2"],
            ],
            3,
        ),
        (
            [
                ["", "Total", "2020", "2021-2022"],
                ["Debt", "$111", "$2", "$5"],
                ["Leases", "1", "2", "3"],
            ],
            1,
        ),
        (
            [
                ["", "Less than", "1 - 3"],
                ["", "1 year", "years"],
                ["Debt", "$17", "$98"],
                ["X", "1", "2"],
            ],
            2,
        ),
        ([["", "2019", "2018"], ["Current:", "", ""], ["Cash", "5", "4"]], 1),
        ([["", "Q1", "Q2"], ["Sales", "—", "n/a"], ["Cost", "1", "2"]], 1),
    ],
)
def test_tatqa_header_row_count_synthetic(rows: list[list[str]], expected: int) -> None:
    assert tatqa.header_row_count(rows) == expected


def test_tatqa_rejects_duplicate_contexts() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        tatqa.build({"train": [tatqa_context()], "dev": [tatqa_context()]})


def financebench_row(**updates: Any) -> dict[str, Any]:
    row = {
        "financebench_id": "financebench_id_00001",
        "company": "Acme Corp",
        "doc_name": "ACME_2022_10K",
        "question_type": "metrics-generated",
        "question_reasoning": "Information extraction",
        "domain_question_num": None,
        "question": "What was FY2022 capex?",
        "answer": "$1,577.00",
        "justification": "Cash flow statement.",
        "dataset_subset_label": "OPEN_SOURCE",
        "evidence": [
            {
                "evidence_text": "Capex 1,577",
                "doc_name": "ACME_2022_10K",
                "evidence_page_num": 59,
                "evidence_text_full_page": " Cash flows\nCapex 1,577 \n",
            }
        ],
        "gics_sector": "Industrials",
        "doc_type": "10k",
        "doc_period": 2022,
        "doc_link": "https://example.test/acme-2022.pdf",
    }
    row.update(updates)
    return row


def test_financebench_is_eval_only_with_one_based_page_documents() -> None:
    documents, items = financebench.build([financebench_row()])
    (document,) = documents
    assert document.id == "ACME_2022_10K/page_60"
    assert document.title == "Acme Corp 10K 2022, page 60"
    assert document.text == "Cash flows\nCapex 1,577"
    assert document.allowed_groups == ["company:Acme Corp"]
    assert document.source == "https://example.test/acme-2022.pdf"
    (item,) = items
    assert (item.license, item.usage, item.split) == ("CC-BY-NC-4.0", "eval_only", "open_source")
    assert item.gold_document_ids == ["ACME_2022_10K/page_60"]
    assert item.gold_unit_ids == ["ACME_2022_10K/page_60#page"]
    assert item.answer_type == "number"
    assert "Capex 1,577" not in document.title and item.answer not in document.text


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("24.26", "number"), ("$8.70", "number"), ("65.4%", "percent"), ("(1.2)", "number")],
)
def test_financebench_answer_type(answer: str, expected: str) -> None:
    assert financebench.answer_type(answer) == expected


def test_financebench_free_text_and_conflicts() -> None:
    assert financebench.answer_type("No, margins declined from 36.8% to 34.6%.") == "text"
    conflicting = financebench_row(financebench_id="other")
    conflicting["evidence"] = [{**conflicting["evidence"][0], "evidence_text_full_page": "x"}]
    with pytest.raises(ValueError, match="Conflicting"):
        financebench.build([financebench_row(), conflicting])
    foreign = financebench_row()
    foreign["evidence"] = [{**foreign["evidence"][0], "doc_name": "OTHER_2022_10K"}]
    with pytest.raises(ValueError, match="another filing"):
        financebench.build([foreign])


def test_corpus_jsonl_is_directly_ingestible_and_round_trips(tmp_path: Path) -> None:
    documents, items = financebench.build([financebench_row()])
    write_jsonl(tmp_path / "documents.jsonl", documents)
    write_jsonl(tmp_path / "questions.jsonl", items)
    (ingestible,) = list(jsonl_documents(tmp_path / "documents.jsonl"))
    assert ingestible == documents[0].to_document()
    assert read_jsonl(tmp_path / "questions.jsonl", QAItem) == items
    assert read_jsonl(tmp_path / "documents.jsonl", CorpusDocument) == documents


def test_build_is_deterministic_and_reports_measured_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "financebench_merged.jsonl").write_text(json.dumps(financebench_row()) + "\n")
    monkeypatch.setattr(build, "verify", lambda directory, dataset: None)
    first = build.run("financebench", raw, tmp_path / "a", None)
    second = build.run("financebench", raw, tmp_path / "b", None)
    assert first["outputs"] == second["outputs"]
    assert (first["documents"], first["questions"], first["chunks"]) == (1, 1, 1)
    assert first["usage"] == "eval_only"
    assert first["splits"]["open_source"]["answer_type:number"] == 1
    assert json.loads((tmp_path / "a" / "stats.json").read_text()) == first
