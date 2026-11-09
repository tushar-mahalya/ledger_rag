"""Licensed full FinanceBench split: retrieval-only cases from user-provided files.

The open 150-question sample is handled by ``financebench``. This adapter accepts a
split the user is licensed to use, maps each question to an independently prepared
full-filing corpus and never indexes gold evidence or answers.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Mapping(BaseModel):
    document_ids: dict[str, str]
    corpus_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def normalize(
    input_path: Path, corpus_path: Path, mapping_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mapping = Mapping.model_validate_json(mapping_path.read_text())
    corpus_hash = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    if corpus_hash != mapping.corpus_sha256:
        raise ValueError("Corpus hash does not match the independently prepared corpus manifest")
    corpus_ids: set[str] = set()
    for line in corpus_path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("Corpus must contain document records with string identifiers")
        corpus_ids.add(item["id"])
    rows = []
    seen: set[str] = set()
    for line_number, line in enumerate(input_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"Input row {line_number} must be an object")
        identifier = item.get("financebench_id")
        question = item.get("question")
        doc_name = item.get("doc_name")
        if not (
            isinstance(identifier, str)
            and isinstance(question, str)
            and isinstance(doc_name, str)
            and all(value.strip() for value in (identifier, question, doc_name))
        ):
            raise ValueError(
                f"Input row {line_number} lacks FinanceBench identity/question/doc_name"
            )
        if identifier in seen:
            raise ValueError(f"Duplicate FinanceBench identifier at row {line_number}")
        seen.add(identifier)
        doc_id = mapping.document_ids.get(doc_name)
        if doc_id is None or doc_id not in corpus_ids:
            raise ValueError(
                f"Input row {line_number} does not map to an independently indexed filing"
            )
        rows.append(
            {
                "id": identifier,
                "question": question,
                "relevant_document_ids": [doc_id],
                "should_abstain": False,
            }
        )
    if not rows:
        raise ValueError("Input split is empty")
    manifest = {
        "dataset": "FinanceBench",
        "rows": len(rows),
        "adapter_version": 1,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "corpus_sha256": corpus_hash,
        "mapping_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
        "protocol": (
            "Full independently prepared filings only. Gold evidence and answers are not used "
            "for corpus construction or retrieval. Output labels measure document retrieval, "
            "not answer correctness or evidence-page localization."
        ),
        "terms": "User must verify benchmark and underlying filing rights; no dataset downloaded.",
    }
    return rows, manifest
