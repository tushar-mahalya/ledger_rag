"""Leakage-safe retriever training data built from FinQA.

* ``train`` questions produce (query, positive, hard negative) triplets.
* ``dev`` questions produce the IR set used for model selection and promotion.
* ``test`` questions are never used for training or selection. Their pages are also
  excluded from negative mining, so the model never learns to push held-out evidence away.

Queries and passages are rendered exactly as the production retriever renders them
(raw question, ``contextual_text(chunk)``), so a fine-tuned model trains on the same
distribution it serves.
"""

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from ledger_rag.core.models import Chunk
from ledger_rag.evaluation.benchmarks.finqa import Case, Corpus
from ledger_rag.ingestion.chunking import chunk_document, contextual_text
from ledger_rag.retrieval.memory import tokenize

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class Triplet:
    query_id: str
    anchor: str
    positive: str
    negative: str


@dataclass
class IRSet:
    queries: dict[str, str] = field(default_factory=dict)
    corpus: dict[str, str] = field(default_factory=dict)
    relevant: dict[str, set[str]] = field(default_factory=dict)


@dataclass
class MiningStats:
    queries: int = 0
    skipped_without_gold: int = 0
    hard_negatives: int = 0
    random_negatives: int = 0
    filtered_false_negatives: int = 0


def corpus_chunks(corpus: Corpus) -> dict[str, list[Chunk]]:
    return {doc_id: chunk_document(doc) for doc_id, doc in sorted(corpus.documents.items())}


def gold_chunks(case: Case, chunks: list[Chunk]) -> list[Chunk]:
    """Chunks of the gold page that contain at least one annotated evidence unit."""
    return [c for c in chunks if any(unit in c.text for unit in case.gold_units)]


def _numbers(text: str) -> set[str]:
    # Short integers ("1", "12") are too common in tables to signal shared evidence.
    values = {m.group().replace(",", "") for m in _NUMBER.finditer(text)}
    return {v for v in values if len(v.replace(".", "")) >= 3}


def is_false_negative(text: str, case: Case) -> bool:
    """Reject candidates that restate the gold evidence (e.g. prior-year comparatives).

    Filings repeat last year's figures, so a page from another year often contains the
    exact numbers that answer the question. Training on it as a negative would teach the
    model to suppress correct evidence.
    """
    if any(unit in text for unit in case.gold_units):
        return True
    gold: set[str] = set()
    for unit in case.gold_units:
        gold |= _numbers(unit)
    return bool(gold) and len(gold & _numbers(text)) / len(gold) >= 0.5


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.2, b: float = 0.75) -> None:
        self._docs = [Counter(d) for d in documents]
        self._lengths = [len(d) for d in documents]
        self._avg = (sum(self._lengths) / len(documents)) if documents else 0.0
        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(doc.keys())
        n = len(documents)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._k1, self._b = k1, b

    def scores(self, query: Iterable[str]) -> list[float]:
        terms = [t for t in set(query) if t in self._idf]
        out: list[float] = []
        for doc, length in zip(self._docs, self._lengths, strict=True):
            norm = self._k1 * (1 - self._b + self._b * length / (self._avg or 1.0))
            out.append(
                sum(
                    self._idf[t] * doc[t] * (self._k1 + 1) / (doc[t] + norm)
                    for t in terms
                    if t in doc
                )
            )
        return out


def build_triplets(
    corpus: Corpus,
    cases: list[Case],
    *,
    excluded_documents: set[str],
    negatives_per_query: int = 3,
    positives_per_query: int = 2,
    seed: int = 13,
) -> tuple[list[Triplet], MiningStats]:
    """Mine same-company, other-page BM25 hard negatives with false-negative filtering."""
    if negatives_per_query < 1 or positives_per_query < 1:
        raise ValueError("negatives_per_query and positives_per_query must be >= 1")
    rng = random.Random(seed)
    chunks = corpus_chunks(corpus)
    by_company: dict[str, list[Chunk]] = defaultdict(list)
    for doc_id, items in chunks.items():
        if doc_id not in excluded_documents:
            by_company[doc_id.split("/")[0]].extend(items)
    indexes = {
        company: BM25([tokenize(contextual_text(c)) for c in items])
        for company, items in by_company.items()
    }
    pool = sorted((c for items in by_company.values() for c in items), key=lambda c: c.id)
    stats = MiningStats()
    triplets: list[Triplet] = []
    for case in cases:
        if case.document_id in excluded_documents:
            raise ValueError(f"Training case {case.id} uses an excluded (held-out) page")
        positives = gold_chunks(case, chunks.get(case.document_id, []))[:positives_per_query]
        if not positives:
            stats.skipped_without_gold += 1
            continue
        stats.queries += 1
        candidates = by_company.get(case.company, [])
        negatives: list[Chunk] = []
        if candidates:
            scores = indexes[case.company].scores(tokenize(case.question))
            order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i].id))
            for i in order:
                chunk = candidates[i]
                if chunk.document_id == case.document_id:
                    continue
                if is_false_negative(chunk.text, case):
                    stats.filtered_false_negatives += 1
                    continue
                negatives.append(chunk)
                if len(negatives) == negatives_per_query:
                    break
        stats.hard_negatives += len(negatives)
        attempts = 0
        while len(negatives) < negatives_per_query and pool and attempts < 100:
            attempts += 1
            chunk = pool[rng.randrange(len(pool))]
            if chunk.document_id != case.document_id and not is_false_negative(chunk.text, case):
                negatives.append(chunk)
                stats.random_negatives += 1
        for positive in positives:
            for negative in negatives:
                triplets.append(
                    Triplet(
                        query_id=case.id,
                        anchor=case.question,
                        positive=contextual_text(positive),
                        negative=contextual_text(negative),
                    )
                )
    return triplets, stats


def build_ir_set(corpus: Corpus, cases: list[Case]) -> IRSet:
    """Full-corpus retrieval task: every chunk is a candidate; gold chunks are relevant."""
    chunks = corpus_chunks(corpus)
    ir = IRSet(corpus={c.id: contextual_text(c) for items in chunks.values() for c in items})
    for case in cases:
        relevant = {c.id for c in gold_chunks(case, chunks.get(case.document_id, []))}
        if relevant:
            ir.queries[case.id] = case.question
            ir.relevant[case.id] = relevant
    return ir


def fingerprint(rows: Iterable[Any]) -> str:
    """Order-sensitive SHA-256 over dataclass rows for dataset lineage."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(asdict(row), sort_keys=True).encode())
        digest.update(b"\n")
    return digest.hexdigest()
