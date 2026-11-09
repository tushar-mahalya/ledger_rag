import math
import re
from collections import Counter

from ledger_rag.core.config import Settings
from ledger_rag.core.contracts import Retriever
from ledger_rag.core.models import Chunk, Hit, Principal, QueryRequest


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def authorized(chunk: Chunk, principal: Principal, query: QueryRequest) -> bool:
    return (
        chunk.tenant_id == principal.tenant_id
        and (not chunk.allowed_groups or bool(set(chunk.allowed_groups) & set(principal.groups)))
        and (query.as_of is None or chunk.published_at <= query.as_of)
    )


class MemoryRetriever:
    """Deterministic lexical demo. Not a substitute for a production vector index."""

    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}

    async def upsert(self, chunks: list[Chunk]) -> None:
        replacements = {(c.tenant_id, c.document_id) for c in chunks}
        self.chunks = {
            key: chunk
            for key, chunk in self.chunks.items()
            if (chunk.tenant_id, chunk.document_id) not in replacements
        }
        self.chunks.update({chunk.id: chunk for chunk in chunks})

    async def update_document_acl(
        self, tenant_id: str, document_id: str, allowed_groups: list[str]
    ) -> int:
        changed = 0
        for key, chunk in list(self.chunks.items()):
            if chunk.tenant_id == tenant_id and chunk.document_id == document_id:
                self.chunks[key] = chunk.model_copy(update={"allowed_groups": list(allowed_groups)})
                changed += 1
        return changed

    async def delete_document(self, tenant_id: str, document_id: str) -> int:
        doomed = [
            key
            for key, chunk in self.chunks.items()
            if chunk.tenant_id == tenant_id and chunk.document_id == document_id
        ]
        for key in doomed:
            del self.chunks[key]
        return len(doomed)

    async def search(self, query: QueryRequest, principal: Principal) -> list[Hit]:
        corpus = [chunk for chunk in self.chunks.values() if authorized(chunk, principal, query)]
        if not corpus:
            return []
        tokens = [Counter(tokenize(f"{chunk.title} {chunk.text}")) for chunk in corpus]
        terms = set(tokenize(query.question))
        lengths = [sum(counts.values()) for counts in tokens]
        average = sum(lengths) / len(lengths) or 1
        frequencies = {term: sum(1 for counts in tokens if term in counts) for term in terms}
        hits: list[Hit] = []
        for chunk, counts, length in zip(corpus, tokens, lengths, strict=True):
            score = 0.0
            for term in terms:
                frequency = counts[term]
                if frequency:
                    idf = math.log(
                        1 + (len(corpus) - frequencies[term] + 0.5) / (frequencies[term] + 0.5)
                    )
                    score += (
                        idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / average))
                    )
            if score > 0:
                hits.append(Hit(chunk=chunk, score=score))
        hits.sort(key=lambda hit: (-hit.score, hit.chunk.id))
        return hits[: query.top_k]

    async def ready(self) -> bool:
        return True

    async def close(self) -> None:
        return None


def make_retriever(settings: Settings) -> Retriever:
    store: Retriever
    if settings.retrieval_backend == "qdrant":
        from ledger_rag.retrieval.qdrant import QdrantRetriever

        store = QdrantRetriever(settings)
    else:
        store = MemoryRetriever()
    if settings.reranker_url:
        from ledger_rag.retrieval.reranking import RerankingRetriever

        return RerankingRetriever(store, settings)
    return store
