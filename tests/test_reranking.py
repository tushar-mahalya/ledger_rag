import json

import httpx
import pytest

from ledger_rag.chunking import chunk_document
from ledger_rag.config import Settings
from ledger_rag.models import Document, Principal, Profile, QueryRequest
from ledger_rag.reranking import RerankingRetriever
from ledger_rag.retrieval import MemoryRetriever


async def test_reranker_orders_candidates(
    settings: Settings, document: Document, principal: Principal
) -> None:
    store = MemoryRetriever()
    other = document.model_copy(update={"id": "second", "text": "Revenue was different."})
    await store.upsert(chunk_document(document) + chunk_document(other))
    wrapper = RerankingRetriever(store, settings.model_copy(update={"reranker_url": "http://tei"}))

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["query"] == "Revenue"
        assert len(payload["texts"]) == 2
        return httpx.Response(200, json=[{"index": 0, "score": 0.1}, {"index": 1, "score": 0.9}])

    await wrapper.client.aclose()
    wrapper.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    initial = await store.search(QueryRequest(question="Revenue"), principal)
    ranked = await wrapper.search(QueryRequest(question="Revenue", top_k=1), principal)
    assert ranked[0].chunk.id == initial[1].chunk.id
    voice = await wrapper.search(QueryRequest(question="Revenue", profile=Profile.VOICE), principal)
    assert [hit.chunk.id for hit in voice] == [hit.chunk.id for hit in initial]
    await wrapper.close()


@pytest.mark.parametrize(
    "payload",
    [[{"index": -1, "score": 0.2}], [{"index": 0, "score": "bad"}], [], {"index": 0}],
)
async def test_invalid_rerank_fails_explicitly(
    settings: Settings, document: Document, principal: Principal, payload: object
) -> None:
    store = MemoryRetriever()
    await store.upsert(chunk_document(document))
    wrapper = RerankingRetriever(store, settings)
    await wrapper.client.aclose()
    wrapper.client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    )
    with pytest.raises(ValueError):
        await wrapper.search(QueryRequest(question="Revenue"), principal)
    await wrapper.close()
