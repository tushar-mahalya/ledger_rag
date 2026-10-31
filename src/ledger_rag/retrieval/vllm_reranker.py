"""Client for the Qwen3-Reranker vLLM proxy (``compute/reranker.py``).

Same ``Retriever`` interface as ``RerankingRetriever``. Unlike the TEI wrapper,
voice is not forced to bypass reranking: when ``voice_url`` is set (e.g. the 0.6B
endpoint), voice requests use it with ``voice_candidates``. Chat uses ``url`` (e.g. 4B).
"""

import math

import httpx
from opentelemetry import trace

from ledger_rag.core.config import Settings
from ledger_rag.core.contracts import Retriever
from ledger_rag.core.models import Chunk, Hit, Principal, Profile, QueryRequest
from ledger_rag.ingestion.chunking import contextual_text
from ledger_rag.retrieval.memory import authorized

DEFAULT_INSTRUCTION = (
    "Given a question about a company's SEC filing, retrieve the filing passage or table "
    "that contains the facts needed to answer it"
)


class VllmRerankingRetriever:
    """Instruction-aware Qwen3 reranking with optional voice/chat endpoint routing."""

    def __init__(
        self,
        inner: Retriever,
        settings: Settings,
        *,
        voice_url: str = "",
        voice_candidates: int = 20,
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> None:
        if not 1 <= voice_candidates <= 100:
            raise ValueError("voice_candidates must be between 1 and 100")
        if not instruction.strip() or len(instruction) > 500:
            raise ValueError("instruction must be nonempty and at most 500 characters")
        self.inner = inner
        self.url = settings.reranker_url.rstrip("/")
        self.voice_url = voice_url.rstrip("/")
        self.candidates = settings.rerank_candidates
        self.voice_candidates = voice_candidates
        self.instruction = instruction
        key = settings.reranker_api_key
        headers = {"Authorization": f"Bearer {key.get_secret_value()}"} if key else {}
        self.client = httpx.AsyncClient(
            timeout=settings.reranker_timeout_seconds, headers=headers, follow_redirects=False
        )

    def _route(self, query: QueryRequest) -> tuple[str, int] | None:
        if query.profile == Profile.VOICE:
            return (self.voice_url, self.voice_candidates) if self.voice_url else None
        return self.url, self.candidates

    async def search(self, query: QueryRequest, principal: Principal) -> list[Hit]:
        route = self._route(query)
        if route is None:
            return await self.inner.search(query, principal)
        url, count = route
        candidates = await self.inner.search(
            query.model_copy(update={"top_k": max(count, query.top_k)}), principal
        )
        if any(not authorized(hit.chunk, principal, query) for hit in candidates):
            raise RuntimeError("Unauthorized reranking candidates")
        if not candidates:
            return []
        with trace.get_tracer(__name__).start_as_current_span("rag.rerank") as span:
            span.set_attribute("rag.rerank.candidates", len(candidates))
            span.set_attribute("rag.rerank.profile", query.profile.value)
            response = await self.client.post(
                f"{url}/rerank",
                json={
                    "query": query.question,
                    "texts": [contextual_text(hit.chunk) for hit in candidates],
                    "instruction": self.instruction,
                },
            )
            response.raise_for_status()
            data = response.json()
        return _rank(candidates, data)[: query.top_k]

    async def upsert(self, chunks: list[Chunk]) -> None:
        await self.inner.upsert(chunks)

    async def ready(self) -> bool:
        return await self.inner.ready()

    async def close(self) -> None:
        await self.client.aclose()
        await self.inner.close()


def _rank(candidates: list[Hit], data: object) -> list[Hit]:
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
    return ranked
