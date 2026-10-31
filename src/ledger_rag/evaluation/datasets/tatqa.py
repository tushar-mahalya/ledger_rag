"""TAT-QA adapter: hybrid table + paragraph contexts from real financial reports.

Every upstream context (one table plus its paragraphs) becomes one document with id
``tatqa/<table uid>``. TAT-QA carries no company names, filing identities or dates,
so ``company`` is unknown and ``fiscal_year`` is the latest year found in the table
header (``None`` when absent). ``published_at`` approximates FinQA's convention
(31 December of that year) and falls back to ``UNKNOWN_DATE`` so date filters never
exclude an undated context. Gold answers and derivations stay in ``QAItem`` only.
"""

import re
from collections.abc import Iterator, Mapping
from datetime import date
from typing import Any

from ledger_rag.evaluation.datasets.schema import (
    AnswerType,
    CorpusDocument,
    QAItem,
    Unit,
    format_number,
    render_table,
    table_rows,
    unit_id,
)
from ledger_rag.evaluation.datasets.sources import SOURCES

SOURCE = SOURCES["tatqa"]
TENANT = "tatqa"
SPLIT_FILES = {
    "train": "tatqa_dataset_train.json",
    "dev": "tatqa_dataset_dev.json",
    "test": "tatqa_dataset_test_gold.json",
}
UNKNOWN_DATE = date(1970, 1, 1)
_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
_NUMERIC = re.compile(r"^[\s$€£(−-]*\d[\d,.\s]*%?\)?$")
_TYPES: dict[str, AnswerType] = {
    "span": "span",
    "multi-span": "multi_span",
    "arithmetic": "number",
    "count": "number",
}


def header_row_count(rows: list[list[str]]) -> int:
    """Leading rows whose value cells are labels or years (at least one row).

    A label-only row (e.g. ``Current:`` with blank values) after the header starts
    the body; it is a section heading, not a column label.
    """
    count = 0
    for row in rows[:-1]:
        values = [cell for cell in row[1:] if cell]
        if not values and count:
            break
        if any(_NUMERIC.match(cell) and not _YEAR.fullmatch(cell) for cell in values):
            break
        count += 1
    return max(1, count)


def fiscal_year(rows: list[list[str]], header_rows: int) -> int | None:
    years = [int(y) for row in rows[:header_rows] for cell in row for y in _YEAR.findall(cell)]
    return max(years) if years else None


def corpus_document(context: Mapping[str, Any], split: str) -> CorpusDocument:
    table_uid = context["table"]["uid"]
    document_id = f"tatqa/{table_uid}"
    rows = table_rows(context["table"]["table"])
    header_rows = header_row_count(rows)
    year = fiscal_year(rows, header_rows)
    table_text = render_table(rows, header_rows)
    units = [
        Unit(
            unit_id=unit_id(document_id, "table"),
            kind="table",
            text=table_text,
            rows=rows,
            header_rows=header_rows,
        )
    ]
    paragraphs = sorted(context["paragraphs"], key=lambda paragraph: int(paragraph["order"]))
    for paragraph in paragraphs:
        if paragraph["text"].strip():
            units.append(
                Unit(
                    unit_id=unit_id(document_id, f"paragraph_{paragraph['order']}"),
                    kind="text",
                    text=paragraph["text"].strip(),
                )
            )
    body = "\n\n".join(unit.text for unit in units if unit.text)
    return CorpusDocument(
        id=document_id,
        tenant_id=TENANT,
        title=f"Financial report excerpt, fiscal {year}" if year else "Financial report excerpt",
        text=body or ".",
        source=f"{SOURCE.homepage}/blob/{SOURCE.revision}/dataset_raw/{SPLIT_FILES[split]}",
        published_at=date(year, 12, 31) if year else UNKNOWN_DATE,
        dataset="tatqa",
        company=None,
        fiscal_year=year,
        units=units,
    )


def _answer(raw: Any, kind: AnswerType) -> tuple[str, list[str]]:
    if isinstance(raw, list):
        spans = [str(value).strip() for value in raw]
        return " | ".join(spans), spans
    if isinstance(raw, int | float) and not isinstance(raw, bool):
        return format_number(raw), []
    text = str(raw).strip()
    return text, [text] if kind in {"span", "multi_span"} else []


def qa_items(context: Mapping[str, Any], document: CorpusDocument, split: str) -> Iterator[QAItem]:
    units = {unit.unit_id for unit in document.units}
    for question in sorted(context["questions"], key=lambda item: int(item["order"])):
        kind = _TYPES[question["answer_type"]]
        scale = question["scale"] or None
        if kind == "number" and scale == "percent":
            kind = "percent"
        answer, spans = _answer(question["answer"], kind)
        gold = []
        if "table" in question["answer_from"]:
            gold.append(unit_id(document.id, "table"))
        for order in question["rel_paragraphs"]:
            key = unit_id(document.id, f"paragraph_{order}")
            if key in units and key not in gold:
                gold.append(key)
        derivation = question["derivation"].strip()
        yield QAItem(
            id=f"tatqa/{split}/{question['uid']}",
            dataset="tatqa",
            split=split,
            question=question["question"],
            answer=answer,
            answers=spans,
            answer_type=kind,
            scale=scale,
            gold_document_ids=[document.id],
            gold_unit_ids=gold,
            program=derivation if question["answer_type"] == "arithmetic" and derivation else None,
            license=SOURCE.license,
            usage=SOURCE.usage,
            metadata={
                "answer_from": question["answer_from"],
                "source_answer_type": question["answer_type"],
                "req_comparison": bool(question["req_comparison"]),
                "derivation": derivation or None,
                "order": int(question["order"]),
            },
        )


def build(
    raw: Mapping[str, list[dict[str, Any]]],
) -> tuple[list[CorpusDocument], list[QAItem]]:
    documents: list[CorpusDocument] = []
    items: list[QAItem] = []
    seen: set[str] = set()
    for split, contexts in raw.items():
        for context in contexts:
            document = corpus_document(context, split)
            if document.id in seen:
                raise ValueError(f"Duplicate TAT-QA table identity {document.id}")
            seen.add(document.id)
            documents.append(document)
            items.extend(qa_items(context, document, split))
    return documents, items
