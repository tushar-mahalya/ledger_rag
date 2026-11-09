"""Normalized corpus and QA schema shared by every evaluation dataset adapter.

``CorpusDocument`` is a ``Document`` subclass, so its JSONL is directly ingestible by
``ledger_rag.ingestion.bulk.jsonl_documents`` (extra fields are ignored there).
Gold answers, programs and evidence labels live only in ``QAItem`` records and are
never written into indexed document text.
"""

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ledger_rag.core.models import Document

DatasetName = Literal["convfinqa", "tatqa", "financebench"]
AnswerType = Literal["number", "percent", "span", "multi_span", "yes_no", "text"]
Usage = Literal["eval_only", "train_ok"]
MetadataValue = str | int | float | bool | list[str] | None


class Unit(BaseModel):
    """Smallest addressable evidence unit: a sentence/paragraph or a (part of a) table."""

    model_config = ConfigDict(extra="forbid")

    unit_id: str = Field(min_length=1)
    kind: Literal["text", "table"]
    text: str
    # Tables keep their structure: ``rows`` includes the first ``header_rows`` header rows.
    rows: list[list[str]] = Field(default_factory=list)
    header_rows: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def structured_tables(self) -> "Unit":
        if self.kind == "text" and (self.rows or self.header_rows):
            raise ValueError("Text units cannot carry table rows")
        if self.kind == "table" and not self.rows:
            raise ValueError("Table units must keep their rows")
        if self.header_rows > len(self.rows):
            raise ValueError("header_rows exceeds the number of rows")
        return self


class CorpusDocument(Document):
    dataset: DatasetName
    company: str | None = None
    fiscal_year: int | None = None
    units: list[Unit] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_units(self) -> "CorpusDocument":
        ids = [unit.unit_id for unit in self.units]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate unit identity in document")
        return self

    def to_document(self) -> Document:
        return Document.model_validate(self.model_dump(include=set(Document.model_fields)))

    def descriptive_metadata(self) -> dict[str, str]:
        """Dataset, company and fiscal year, for corpus specs that index them as metadata.

        Not applied by ``to_document``: ConvFinQA must stay byte-identical to the FinQA
        corpus it reuses, so metadata is opt-in per corpus spec."""
        metadata: dict[str, str] = {"dataset": self.dataset}
        if self.company:
            metadata["company"] = self.company[:200]
        if self.fiscal_year is not None:
            metadata["fiscal_year"] = str(self.fiscal_year)
        return metadata


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class QAItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    dataset: DatasetName
    split: str = Field(min_length=1)
    turns: list[Turn] = Field(default_factory=list)
    question: str = Field(min_length=1)
    answer: str
    # Individual gold spans for span/multi_span answers; ``answer`` joins them.
    answers: list[str] = Field(default_factory=list)
    answer_type: AnswerType
    scale: str | None = None
    gold_document_ids: list[str] = Field(min_length=1)
    gold_unit_ids: list[str] = Field(default_factory=list)
    program: str | None = None
    license: str
    usage: Usage
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


def unit_id(document_id: str, key: str) -> str:
    return f"{document_id}#{key}"


def _cell(cell: str) -> str:
    return re.sub(r"\s+", " ", cell.replace("|", "/")).strip()


def table_rows(table: list[list[str]]) -> list[list[str]]:
    """Whitespace-normalized rectangular rows (short rows are right-padded)."""
    width = max((len(row) for row in table), default=0)
    return [[_cell(c) for c in row] + [""] * (width - len(row)) for row in table]


def render_table(rows: list[list[str]], header_rows: int = 1) -> str:
    """Markdown table compatible with the table-aware chunker.

    Multi-row headers are merged column-wise into one header row, so every chunk
    continuation (which repeats a single header row) keeps all column labels.
    """
    if not rows:
        return ""
    split = min(max(1, header_rows), len(rows))
    header = [
        " ".join(cell for cell in column if cell) for column in zip(*rows[:split], strict=True)
    ]
    rendered = ["| " + " | ".join(row) + " |" for row in [header, *rows[split:]]]
    separator = "|" + "---|" * len(header)
    return "\n".join([rendered[0], separator, *rendered[1:]])


def format_number(value: float | int) -> str:
    """Stable, compact decimal string for executed numeric answers."""
    if isinstance(value, bool):
        raise TypeError("Booleans are not numeric answers")
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


Model = TypeVar("Model", bound=BaseModel)


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(record.model_dump_json() + "\n")
            count += 1
    temporary.replace(path)
    return count


def read_jsonl(path: Path, model: type[Model]) -> list[Model]:
    with path.open(encoding="utf-8") as stream:
        return [model.model_validate_json(line) for line in stream if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
