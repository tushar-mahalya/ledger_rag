"""ConvFinQA adapter: multi-turn numerical conversations over FinQA report pages.

Upstream represents each conversation once (``train.json``/``dev.json``) with an
``annotation`` holding parallel per-turn lists:

* ``dialogue_break``: the user question of every turn;
* ``turn_program``: the full program that answers that turn (later turns repeat the
  earlier steps, e.g. ``subtract(a, b), divide(#0, b)``);
* ``exe_ans_list``: the executed gold answer of every turn (ratios stay ratios);
* ``qa_split``: which underlying FinQA question (``qa`` or ``qa_0``/``qa_1`` for
  "Double" conversations) the turn decomposes, used here for gold evidence.

Each turn becomes one ``QAItem``. Its ``turns`` field is the teacher-forced history:
every earlier user question followed by the earlier gold executed answer, mirroring
the turn-level upstream files (``*_turn.json``) that expand conversations the same
way. The private test split has no answers and yields documents only.

Pages are FinQA pages: ``corpus_document`` reuses FinQA's own ``page_document`` so
ids, titles, tenant and dates match the ``finqa_gte_modernbert_v1`` collection.
"""

import json
import zipfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from ledger_rag.core.models import Document
from ledger_rag.evaluation.benchmarks.finqa import page_document
from ledger_rag.evaluation.datasets.schema import (
    AnswerType,
    CorpusDocument,
    QAItem,
    Turn,
    Unit,
    format_number,
    render_table,
    table_rows,
    unit_id,
)
from ledger_rag.evaluation.datasets.sources import SOURCES

SOURCE = SOURCES["convfinqa"]
SPLIT_FILES = {"train": "data/train.json", "dev": "data/dev.json", "test": "data/test_private.json"}
LABELED_SPLITS = ("train", "dev")


def load_raw(archive: Path) -> dict[str, list[dict[str, Any]]]:
    with zipfile.ZipFile(archive) as bundle:
        return {split: json.loads(bundle.read(name)) for split, name in SPLIT_FILES.items()}


def page_units(example: Mapping[str, Any], document_id: str) -> list[Unit]:
    """FinQA unit keys: ``text_i`` over pre+post sentences, ``table_i`` per table row."""
    units: list[Unit] = []
    texts = list(example["pre_text"]) + list(example["post_text"])
    for index, text in enumerate(texts):
        if text.strip():
            units.append(
                Unit(unit_id=unit_id(document_id, f"text_{index}"), kind="text", text=text.strip())
            )
    rows = table_rows(example["table"])
    for index, row in enumerate(rows):
        structured = [row] if index == 0 else [rows[0], row]
        units.append(
            Unit(
                unit_id=unit_id(document_id, f"table_{index}"),
                kind="table",
                text=render_table(structured),
                rows=structured,
                header_rows=1,
            )
        )
    return units


def corpus_document(example: Mapping[str, Any]) -> CorpusDocument:
    document, _ = page_document(dict(example))
    company, year = document.id.split("/")[:2]
    return CorpusDocument(
        **document.model_dump(),
        dataset="convfinqa",
        company=company,
        fiscal_year=int(year),
        units=page_units(example, document.id),
    )


def _gold_units(example: Mapping[str, Any], qa_index: int, units: set[str]) -> list[str]:
    qa = example["qa"] if "qa" in example else example[f"qa_{qa_index}"]
    keys = [unit_id(example["filename"], key) for key in qa.get("gold_inds", {})]
    return [key for key in keys if key in units]


def _answer(value: float | int | str) -> tuple[str, AnswerType]:
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized, "yes_no" if normalized in {"yes", "no"} else "text"
    return format_number(value), "number"


def qa_items(example: Mapping[str, Any], split: str) -> Iterator[QAItem]:
    annotation = example["annotation"]
    questions: list[str] = annotation["dialogue_break"]
    programs: list[str] = annotation["turn_program"]
    answers: list[float | int | str] = annotation["exe_ans_list"]
    qa_split: list[int] = annotation["qa_split"]
    if not len(questions) == len(programs) == len(answers) == len(qa_split):
        raise ValueError(f"ConvFinQA conversation {example['id']} has misaligned turns")
    known_units = {unit.unit_id for unit in page_units(example, example["filename"])}
    kind = "double" if "qa_0" in example else "single"
    history: list[Turn] = []
    for index, question in enumerate(questions):
        answer, answer_type = _answer(answers[index])
        qa = example["qa"] if kind == "single" else example[f"qa_{qa_split[index]}"]
        yield QAItem(
            id=f"convfinqa/{split}/{example['id']}/turn_{index}",
            dataset="convfinqa",
            split=split,
            turns=list(history),
            question=question,
            answer=answer,
            answer_type=answer_type,
            gold_document_ids=[example["filename"]],
            gold_unit_ids=_gold_units(example, qa_split[index], known_units),
            program=programs[index],
            license=SOURCE.license,
            usage=SOURCE.usage,
            metadata={
                "conversation_id": example["id"],
                "conversation_type": kind,
                "turn_index": index,
                "turns_in_conversation": len(questions),
                "qa_split": qa_split[index],
                "finqa_question": qa["question"],
            },
        )
        history += [Turn(role="user", content=question), Turn(role="assistant", content=answer)]


def finqa_coverage(
    raw: Mapping[str, list[dict[str, Any]]], finqa_documents: Mapping[str, Document]
) -> dict[str, Any]:
    """Exact id and content coverage of ConvFinQA pages by the ingested FinQA corpus."""
    report: dict[str, Any] = {"finqa_documents": len(finqa_documents), "splits": {}}
    all_pages: set[str] = set()
    all_missing: set[str] = set()
    all_mismatched: set[str] = set()
    for split, examples in raw.items():
        pages = {example["filename"] for example in examples}
        missing = {page for page in pages if page not in finqa_documents}
        mismatched = {
            example["filename"]
            for example in examples
            if example["filename"] in finqa_documents
            and page_document(example)[0] != finqa_documents[example["filename"]]
        }
        all_pages |= pages
        all_missing |= missing
        all_mismatched |= mismatched
        report["splits"][split] = {
            "conversations": len(examples),
            "unique_pages": len(pages),
            "pages_in_finqa": len(pages) - len(missing),
            "pages_identical_to_finqa": len(pages) - len(missing) - len(mismatched),
        }
    covered = len(all_pages) - len(all_missing)
    report.update(
        unique_pages=len(all_pages),
        pages_in_finqa=covered,
        pages_identical_to_finqa=covered - len(all_mismatched),
        missing_page_ids=sorted(all_missing),
        mismatched_page_ids=sorted(all_mismatched),
        id_coverage=covered / len(all_pages) if all_pages else 0.0,
    )
    return report


def build(
    raw: Mapping[str, list[dict[str, Any]]],
) -> tuple[list[CorpusDocument], list[QAItem]]:
    documents: dict[str, CorpusDocument] = {}
    items: list[QAItem] = []
    for split, examples in raw.items():
        for example in examples:
            if example["filename"] not in documents:
                documents[example["filename"]] = corpus_document(example)
            if split in LABELED_SPLITS:
                items.extend(qa_items(example, split))
    return list(documents.values()), items
