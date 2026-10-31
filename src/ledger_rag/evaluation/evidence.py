"""Deterministic mapping from FinQA gold evidence units to ingested chunks.

FinQA labels supporting facts as ``text_<i>`` (the i-th entry of
``pre_text + post_text``) and ``table_<j>`` (the j-th table row). The benchmark
ingests each report page with :func:`ledger_rag.ingestion.chunking.chunk_document`,
which is deterministic, so chunk IDs and texts can be rebuilt locally without
re-ingestion and matched to retrieved hits by chunk ID.

Mapping rule, per document, after whitespace normalization:

1. A unit maps to every chunk whose text contains the whole unit (``whole``).
2. Otherwise the unit is split with the chunker's own sentence boundary rule and
   each fragment must be contained in some chunk (``fragments``); retrieving the
   unit then requires one containing chunk for every fragment.
3. Otherwise the unit is ``unmapped``: it stays in every denominator, can never be
   retrieved, and is counted in the report. Nothing is silently dropped.

Table rows are matched by their rendered markdown row. Table continuations repeat
header rows, so a header unit can be satisfied by several chunks.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ledger_rag.core.models import Chunk

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _norm(text: str) -> str:
    return " ".join(text.split())


@dataclass(frozen=True)
class UnitMapping:
    key: str
    method: str  # whole | fragments | unmapped
    # One frozenset of acceptable chunk IDs per required fragment.
    requirements: tuple[frozenset[str], ...]

    def satisfied_by(self, chunk_ids: Iterable[str]) -> bool:
        present = set(chunk_ids)
        return bool(self.requirements) and all(req & present for req in self.requirements)

    @property
    def chunk_ids(self) -> frozenset[str]:
        return frozenset().union(*self.requirements) if self.requirements else frozenset()


@dataclass
class DocumentEvidence:
    chunks: list[Chunk]
    units: dict[str, UnitMapping] = field(default_factory=dict)

    @property
    def by_id(self) -> dict[str, Chunk]:
        return {chunk.id: chunk for chunk in self.chunks}


def map_units(units: dict[str, str], chunks: Sequence[Chunk]) -> dict[str, UnitMapping]:
    texts = [(chunk.id, _norm(chunk.text)) for chunk in chunks]
    mapped: dict[str, UnitMapping] = {}
    for key, raw in units.items():
        unit = _norm(raw)
        whole = frozenset(cid for cid, text in texts if unit and unit in text)
        if whole:
            mapped[key] = UnitMapping(key, "whole", (whole,))
            continue
        fragments = [f for f in (_norm(p) for p in _SENTENCE.split(unit)) if f]
        requirements = tuple(
            frozenset(cid for cid, text in texts if fragment in text) for fragment in fragments
        )
        if len(fragments) > 1 and all(requirements):
            mapped[key] = UnitMapping(key, "fragments", requirements)
        else:
            mapped[key] = UnitMapping(key, "unmapped", ())
    return mapped


def evidence_metrics(
    retrieved_ids: Sequence[str],
    gold: Sequence[UnitMapping],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float | None]:
    """Unit recall, chunk precision, all-evidence hit and gold-chunk MRR at cutoffs.

    ``retrieved_ids`` is the ranked chunk-ID list. Metrics are ``None`` (ineligible,
    and counted as such) when the case has no gold units.
    """
    out: dict[str, float | None] = {}
    if not gold:
        for k in ks:
            for name in ("evidence_recall", "evidence_precision", "all_evidence"):
                out[f"{name}@{k}"] = None
        out["evidence_mrr@10"] = None
        return out
    relevant = frozenset().union(*(unit.chunk_ids for unit in gold))
    for k in ks:
        top = list(retrieved_ids[:k])
        found = sum(unit.satisfied_by(top) for unit in gold)
        out[f"evidence_recall@{k}"] = found / len(gold)
        out[f"all_evidence@{k}"] = float(found == len(gold))
        out[f"evidence_precision@{k}"] = (
            sum(cid in relevant for cid in top) / len(top) if top else 0.0
        )
    rank = next((i for i, cid in enumerate(retrieved_ids[:10], 1) if cid in relevant), None)
    out["evidence_mrr@10"] = 1 / rank if rank else 0.0
    return out


def citation_metrics(
    cited_ids: Sequence[str], gold: Sequence[UnitMapping]
) -> dict[str, float | None]:
    """Precision of cited chunks and recall of gold units covered by citations."""
    if not gold:
        return {"citation_precision": None, "citation_recall": None}
    relevant = frozenset().union(*(unit.chunk_ids for unit in gold))
    unique = list(dict.fromkeys(cited_ids))
    return {
        "citation_precision": (
            sum(cid in relevant for cid in unique) / len(unique) if unique else None
        ),
        "citation_recall": sum(unit.satisfied_by(unique) for unit in gold) / len(gold),
    }
