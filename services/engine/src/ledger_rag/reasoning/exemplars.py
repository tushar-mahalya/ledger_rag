"""Retrieve solved training problems as dynamic in-context exemplars.

Similar questions share their program shape ("percentage change from Y1 to Y2" is
``subtract, divide``), so showing the nearest solved problems teaches a small reader
the dataset's conventions (ratio versus ``* 100``, sign, operand order) far better
than fixed examples (Liu et al., 2022, "What Makes Good In-Context Examples";
FINDER, Khatuya et al., EMNLP 2025). Retrieval is BM25 over question text: no model,
deterministic, and built only from the training split, so dev/test never leak.
"""

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z]+|\d+(?:\.\d+)?")
_STOP = frozenset(
    "the a an of in to for and or is was were what how much by as on at from with that this "
    "be are its it their than".split()
)


def tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOP]


@dataclass(frozen=True)
class Exemplar:
    id: str
    question: str
    facts: tuple[str, ...]
    program: str
    group: str = ""
    """Problems sharing a group (same report page) are never shown for each other."""


class ExemplarIndex:
    """Okapi BM25 (k1=1.2, b=0.75) over exemplar questions."""

    def __init__(self, exemplars: Iterable[Exemplar], *, k1: float = 1.2, b: float = 0.75):
        self.exemplars = list(exemplars)
        self._docs = [Counter(tokenize(e.question)) for e in self.exemplars]
        lengths = [sum(doc.values()) for doc in self._docs]
        self._avg = sum(lengths) / len(lengths) if lengths else 0.0
        self._lengths = lengths
        frequency: Counter[str] = Counter()
        for doc in self._docs:
            frequency.update(doc.keys())
        total = len(self._docs)
        self._idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in frequency.items()
        }
        self._postings: dict[str, list[int]] = {}
        for index, doc in enumerate(self._docs):
            for term in doc:
                self._postings.setdefault(term, []).append(index)
        self._k1, self._b = k1, b

    def search(
        self, question: str, k: int, *, exclude_group: str = "", exclude_id: str = ""
    ) -> list[Exemplar]:
        if k <= 0 or not self.exemplars:
            return []
        scores: dict[int, float] = {}
        for term in set(tokenize(question)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for index in self._postings[term]:
                tf = self._docs[index][term]
                norm = 1 - self._b + self._b * self._lengths[index] / self._avg
                scores[index] = scores.get(index, 0.0) + idf * tf * (self._k1 + 1) / (
                    tf + self._k1 * norm
                )
        ranked = sorted(scores, key=lambda index: (-scores[index], index))
        chosen: list[Exemplar] = []
        for index in ranked:
            exemplar = self.exemplars[index]
            if exemplar.id == exclude_id or (exclude_group and exemplar.group == exclude_group):
                continue
            chosen.append(exemplar)
            if len(chosen) == k:
                break
        return chosen


def render(exemplars: Sequence[Exemplar]) -> str:
    """Exemplars in the same FACTS / PROGRAM format the reader must produce."""
    blocks = []
    for exemplar in exemplars:
        facts = "\n".join(f"- {fact}" for fact in exemplar.facts) or "- (none)"
        blocks.append(
            f"Question: {exemplar.question}\nFACTS:\n{facts}\nPROGRAM: {exemplar.program}"
        )
    return "\n\n".join(blocks)
