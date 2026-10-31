import math

import httpx
from opentelemetry import trace

from ledger_rag.core.config import Settings
from ledger_rag.core.contracts import Retriever
from ledger_rag.core.models import Chunk, Hit, Principal, Profile, QueryRequest
from ledger_rag.ingestion.chunking import contextual_text
from ledger_rag.retrieval.memory import authorized


class RerankingRetriever:
    """TEI cross-encoder reranking; voice explicitly bypasses this latency-heavy stage."""

    def __init__(self, inner: Retriever, settings: Settings) -> None:
        self.inner = inner
        self.url = settings.reranker_url.rstrip("/")
        self.candidates = settings.rerank_candidates
        headers = (
            {"Authorization": f"Bearer {settings.reranker_api_key.get_secret_value()}"}
            if settings.reranker_api_key
            else {}
        )
        self.client = httpx.AsyncClient(
            timeout=settings.reranker_timeout_seconds, headers=headers, follow_redirects=False
        )

    async def search(self, query: QueryRequest, principal: Principal) -> list[Hit]:
        if query.profile == Profile.VOICE:
            return await self.inner.search(query, principal)
        candidates = await self.inner.search(
            query.model_copy(update={"top_k": max(self.candidates, query.top_k)}), principal
        )
        if any(not authorized(hit.chunk, principal, query) for hit in candidates):
            raise RuntimeError("Unauthorized reranking candidates")
        if not candidates:
            return []
        with trace.get_tracer(__name__).start_as_current_span("rag.rerank"):
            response = await self.client.post(
                f"{self.url}/rerank",
                json={
                    "query": query.question,
                    "texts": [contextual_text(hit.chunk) for hit in candidates],
                    "truncate": True,
                    "raw_scores": False,
                    "return_text": False,
                },
            )
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, list) or len(data) != len(candidates):
            raise ValueError("Reranker must score all candidates")
        seen: set[int] = set()
        ranked: list[Hit] = []
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Invalid reranker result")
            index, score = item.get("index"), item.get("score")
            if (
                type(index) is not int
                or index < 0
                or index >= len(candidates)
                or index in seen
                or not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(score)
            ):
                raise ValueError("Invalid reranker index or score")
            seen.add(index)
            ranked.append(Hit(chunk=candidates[index].chunk, score=float(score)))
        ranked.sort(key=lambda hit: (-hit.score, hit.chunk.id))
        return ranked[: query.top_k]

    async def upsert(self, chunks: list[Chunk]) -> None:
        await self.inner.upsert(chunks)

    async def ready(self) -> bool:
        return await self.inner.ready()

    async def bootstrap(self) -> None:
        bootstrap = getattr(self.inner, "bootstrap", None)
        if bootstrap is not None:
            await bootstrap()

    async def close(self) -> None:
        await self.client.aclose()
        await self.inner.close()
