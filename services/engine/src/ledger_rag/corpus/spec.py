"""Declarative corpus specifications (``config/corpora/<id>.toml``).

A spec names one source and fixes everything that defines the resulting index: the
owning tenant, the access policy, chunking, the embedding space and the serving
alias. Every build of a spec lands in a new physical collection
``<alias>__<build_id>``; serving reads the alias, so promotion and rollback are one
atomic alias swap.
"""

import secrets
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ledger_rag.ingestion.chunking import ChunkProfile

Classification = Literal["public", "internal", "confidential", "restricted"]
MetadataField = Literal["dataset", "company", "fiscal_year"]
NAME = r"^[a-z][a-z0-9_]{0,47}$"
# 1-256 printable characters, inner spaces allowed (e.g. "company:American Express").
GROUP = r"^[^\s\x00-\x1f](?:[^\x00-\x1f]{0,254}[^\s\x00-\x1f])?$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceSpec(_Strict):
    kind: Literal["dataset", "jsonl"]
    # Dataset registry name (kind = dataset).
    name: str = Field(default="", max_length=64)
    # JSONL of documents, relative to the data root (kind = jsonl).
    path: str = Field(default="", max_length=512)

    @model_validator(mode="after")
    def complete(self) -> "SourceSpec":
        if self.kind == "dataset" and (not self.name or self.path):
            raise ValueError("A dataset source needs `name` and no `path`")
        if self.kind == "jsonl":
            if not self.path or self.name:
                raise ValueError("A jsonl source needs `path` and no `name`")
            if Path(self.path).is_absolute() or ".." in Path(self.path).parts:
                raise ValueError("A jsonl path must be relative to the data root")
        return self


class AccessSpec(_Strict):
    # Groups given to documents that carry none. Empty means tenant-wide visibility.
    default_groups: list[str] = Field(default_factory=list, max_length=100)
    classification: Classification = "internal"

    @field_validator("default_groups")
    @classmethod
    def valid_groups(cls, value: list[str]) -> list[str]:
        import re

        if any(not re.fullmatch(GROUP, group) for group in value):
            raise ValueError("Access groups must be 1-256 printable characters, no edge spaces")
        return value

    @property
    def requires_acl(self) -> bool:
        """Confidential and restricted data may never be visible tenant-wide."""
        return self.classification in {"confidential", "restricted"}


class ProcessingSpec(_Strict):
    chunk_profile: ChunkProfile = "v1"
    # Descriptive fields copied onto documents and chunks (normalized datasets only).
    metadata: list[MetadataField] = Field(default_factory=list)


class EmbeddingSpec(_Strict):
    model: str = Field(pattern=r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$", max_length=180)
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    dimensions: int = Field(ge=32, le=8192)
    quantization: Literal["none", "scalar", "binary"] = "none"

    @property
    def space_id(self) -> str:
        """Same identity the TEI ingestion job records for its float16 runtime."""
        return f"tei-1.8-float16:{self.model}@{self.revision}"


class IndexSpec(_Strict):
    alias: str = Field(pattern=NAME)
    keep_builds: int = Field(default=2, ge=1, le=10)


class CorpusSpec(_Strict):
    id: str = Field(pattern=NAME)
    description: str = Field(min_length=1, max_length=300)
    tenant: str = Field(min_length=1, max_length=200)
    source: SourceSpec
    access: AccessSpec = AccessSpec()
    processing: ProcessingSpec = ProcessingSpec()
    embedding: EmbeddingSpec
    index: IndexSpec

    def build_collection(self, build_id: str) -> str:
        return f"{self.index.alias}__{build_id}"


def new_build_id(now: datetime | None = None) -> str:
    """Sortable, collision-resistant build identifier, e.g. ``20261109t120501_a1b2c3``."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dt%H%M%S")
    return f"{stamp}_{secrets.token_hex(3)}"


def load_spec(path: Path) -> CorpusSpec:
    with path.open("rb") as stream:
        spec = CorpusSpec.model_validate(tomllib.load(stream))
    if spec.id != path.stem:
        raise ValueError(f"Corpus id {spec.id!r} must match its file name {path.name!r}")
    return spec


def spec_path(name_or_path: str, root: Path) -> Path:
    """Resolve ``finqa`` to ``<root>/config/corpora/finqa.toml``; accept explicit paths."""
    candidate = Path(name_or_path)
    if candidate.suffix == ".toml":
        return candidate
    return root / "config" / "corpora" / f"{name_or_path}.toml"
