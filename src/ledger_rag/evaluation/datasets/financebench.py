"""FinanceBench open-source sample (150 questions) adapter. CC-BY-NC-4.0: EVALUATION ONLY.

The pinned Hugging Face file embeds, for every evidence item, the full text of the
evidence page (``evidence_text_full_page``) with a zero-indexed ``evidence_page_num``.
Those pages form an **evidence-page pool**, not FinanceBench's full-filing setting:
every indexed page is gold for at least one question, so retrieval over this pool is
an optimistic, page-localization measurement. Full filings require the 368 PDFs in
the maintainer repository plus PDF extraction (see docs/DATASETS.md).

Document ids are ``<doc_name>/page_<n>`` with a one-based ``n`` (PDF page numbering,
like FinQA's ``page_<n>.pdf``). Gold answers, justifications and evidence spans stay
in ``QAItem`` only; only full page text is indexed.
"""

import re
from collections.abc import Mapping
from datetime import date
from typing import Any

from ledger_rag.evaluation.datasets.schema import (
    AnswerType,
    CorpusDocument,
    QAItem,
    Unit,
    unit_id,
)
from ledger_rag.evaluation.datasets.sources import SOURCES

SOURCE = SOURCES["financebench"]
TENANT = "financebench"
SPLIT = "open_source"
_NUMBER = re.compile(r"^\(?[-−]?\$?\s?\d[\d,]*(?:\.\d+)?\)?\s*(%|percent)?$", re.IGNORECASE)


def page_id(doc_name: str, zero_based_page: int) -> str:
    return f"{doc_name}/page_{zero_based_page + 1}"


def answer_type(answer: str) -> AnswerType:
    match = _NUMBER.match(answer.strip())
    if match:
        return "percent" if match.group(1) else "number"
    return "text"


def _document(row: Mapping[str, Any], evidence: Mapping[str, Any]) -> CorpusDocument:
    page = int(evidence["evidence_page_num"])
    document_id = page_id(evidence["doc_name"], page)
    company = row["company"]
    period = int(row["doc_period"])
    text = evidence["evidence_text_full_page"].strip()
    return CorpusDocument(
        id=document_id,
        tenant_id=TENANT,
        title=f"{company} {row['doc_type'].upper()} {period}, page {page + 1}",
        text=text,
        source=row["doc_link"],
        # FinanceBench gives the reporting period only; this is an approximate date.
        published_at=date(period, 12, 31),
        allowed_groups=[f"company:{company}"],
        dataset="financebench",
        company=company,
        fiscal_year=period,
        units=[Unit(unit_id=unit_id(document_id, "page"), kind="text", text=text)],
    )


def build(rows: list[Mapping[str, Any]]) -> tuple[list[CorpusDocument], list[QAItem]]:
    documents: dict[str, CorpusDocument] = {}
    items: list[QAItem] = []
    for row in rows:
        gold: list[str] = []
        for evidence in row["evidence"]:
            if evidence["doc_name"] != row["doc_name"]:
                raise ValueError(f"{row['financebench_id']} cites evidence from another filing")
            document = _document(row, evidence)
            previous = documents.setdefault(document.id, document)
            if previous.text != document.text:
                raise ValueError(f"Conflicting page text for {document.id}")
            if document.id not in gold:
                gold.append(document.id)
        answer = row["answer"].strip()
        items.append(
            QAItem(
                id=f"financebench/{row['financebench_id']}",
                dataset="financebench",
                split=SPLIT,
                question=row["question"],
                answer=answer,
                answer_type=answer_type(answer),
                gold_document_ids=gold,
                gold_unit_ids=[unit_id(document_id, "page") for document_id in gold],
                license=SOURCE.license,
                usage=SOURCE.usage,
                metadata={
                    "financebench_id": row["financebench_id"],
                    "doc_name": row["doc_name"],
                    "company": row["company"],
                    "doc_type": row["doc_type"],
                    "doc_period": int(row["doc_period"]),
                    "doc_link": row["doc_link"],
                    "gics_sector": row.get("gics_sector"),
                    "question_type": row["question_type"],
                    "question_reasoning": row.get("question_reasoning"),
                    "domain_question_num": row.get("domain_question_num"),
                    "justification": row.get("justification") or None,
                    "evidence_texts": [e["evidence_text"] for e in row["evidence"]],
                },
            )
        )
    return list(documents.values()), items
