import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ledger_rag.core.config import Settings
from ledger_rag.core.models import Document, Principal, Profile, QueryRequest
from ledger_rag.ingestion.chunking import chunk_document
from ledger_rag.retrieval.memory import MemoryRetriever
from ledger_rag.retrieval.vllm_reranker import DEFAULT_INSTRUCTION, VllmRerankingRetriever

ROOT = Path(__file__).resolve().parents[4]  # repository root


def load_reranker() -> ModuleType:
    pytest.importorskip("modal", reason="Modal app tests require compute group")
    spec = importlib.util.spec_from_file_location("ledger_reranker", ROOT / "compute/reranker.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _store(document: Document) -> MemoryRetriever:
    store = MemoryRetriever()
    other = document.model_copy(update={"id": "second", "text": "Revenue was different."})
    await store.upsert(chunk_document(document) + chunk_document(other))
    return store


async def test_chat_and_voice_route_to_separate_endpoints(
    settings: Settings, document: Document, principal: Principal
) -> None:
    store = await _store(document)
    wrapper = VllmRerankingRetriever(
        store,
        settings.model_copy(update={"reranker_url": "http://large/"}),
        voice_url="http://small",
        voice_candidates=5,
    )
    hosts: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        hosts.append(request.url.host)
        assert request.url.path == "/rerank"
        assert payload["instruction"] == DEFAULT_INSTRUCTION
        assert len(payload["texts"]) == 2
        return httpx.Response(200, json=[{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}])

    await wrapper.client.aclose()
    wrapper.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    initial = await store.search(QueryRequest(question="Revenue"), principal)
    chat = await wrapper.search(QueryRequest(question="Revenue", top_k=1), principal)
    voice = await wrapper.search(
        QueryRequest(question="Revenue", top_k=1, profile=Profile.VOICE), principal
    )
    assert chat[0].chunk.id == voice[0].chunk.id == initial[1].chunk.id
    assert hosts == ["large", "small"]
    await wrapper.close()


async def test_voice_bypasses_without_voice_endpoint(
    settings: Settings, document: Document, principal: Principal
) -> None:
    store = await _store(document)
    wrapper = VllmRerankingRetriever(store, settings)

    def fail(request: httpx.Request) -> httpx.Response:
        raise AssertionError("voice must not call the reranker")

    await wrapper.client.aclose()
    wrapper.client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
    query = QueryRequest(question="Revenue", profile=Profile.VOICE)
    voice = await wrapper.search(query, principal)
    assert [h.chunk.id for h in voice] == [h.chunk.id for h in await store.search(query, principal)]
    await wrapper.close()


@pytest.mark.parametrize(
    "payload",
    [[{"index": 0, "score": 1.0}], [{"index": 0, "score": 1}, {"index": 0, "score": 2}], {}],
)
async def test_invalid_scores_fail(
    settings: Settings, document: Document, principal: Principal, payload: object
) -> None:
    wrapper = VllmRerankingRetriever(
        await _store(document), settings.model_copy(update={"reranker_url": "http://large"})
    )
    await wrapper.client.aclose()
    wrapper.client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    )
    with pytest.raises(ValueError):
        await wrapper.search(QueryRequest(question="Revenue"), principal)
    await wrapper.close()


PRODUCTION: dict[str, Any] = {
    "environment": "production",
    "auth_mode": "oidc",
    "oidc_issuer": "https://issuer.example",
    "oidc_audience": "ledger",
    "oidc_jwks_url": "https://issuer.example/jwks",
    "retrieval_backend": "qdrant",
    "embedding_api_key": SecretStr("k"),
    "provider": "openai",
    "llm_api_key": SecretStr("k"),
    "llm_model": "m",
    "reranker_url": "https://large.example",
}


def test_production_rejects_plain_http_voice_endpoint() -> None:
    production = Settings(**PRODUCTION, reranker_voice_url="https://small.example")
    with pytest.raises(ValueError, match="must use HTTPS"):
        Settings(**PRODUCTION, reranker_voice_url="http://small.internal")
    with pytest.raises(ValueError, match="must use HTTPS"):
        VllmRerankingRetriever(MemoryRetriever(), production, voice_url="http://small.internal")
    with pytest.raises(ValueError, match="embedded credentials"):
        VllmRerankingRetriever(MemoryRetriever(), production, voice_url="ftp://small.example")
    wrapper = VllmRerankingRetriever(MemoryRetriever(), production)
    assert wrapper.voice_url == "https://small.example"


def test_development_still_validates_voice_endpoint_shape(settings: Settings) -> None:
    with pytest.raises(ValueError, match="embedded credentials"):
        Settings(demo_token=SecretStr("test-only-token"), reranker_voice_url="small.internal")
    with pytest.raises(ValueError, match="embedded credentials"):
        VllmRerankingRetriever(MemoryRetriever(), settings, voice_url="https://u:p@small")


async def test_chat_bypasses_reranking_without_chat_endpoint(
    settings: Settings, document: Document, principal: Principal
) -> None:
    store = await _store(document)
    wrapper = VllmRerankingRetriever(store, settings, voice_url="http://small", voice_candidates=5)
    hosts: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(200, json=[{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}])

    await wrapper.client.aclose()
    wrapper.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    query = QueryRequest(question="Revenue")
    chat = await wrapper.search(query, principal)
    assert [h.chunk.id for h in chat] == [h.chunk.id for h in await store.search(query, principal)]
    await wrapper.search(QueryRequest(question="Revenue", profile=Profile.VOICE), principal)
    assert hosts == ["small"]
    await wrapper.close()


def test_rejects_bad_configuration(settings: Settings) -> None:
    with pytest.raises(ValueError):
        VllmRerankingRetriever(MemoryRetriever(), settings, voice_candidates=0)
    with pytest.raises(ValueError):
        VllmRerankingRetriever(MemoryRetriever(), settings, instruction=" ")


def test_vllm_command_pins_revision_and_conversion() -> None:
    module = load_reranker()
    command = module.vllm_command("4b")
    assert command[command.index("--revision") + 1] == module.MODELS["4b"][1]
    overrides = json.loads(command[command.index("--hf-overrides") + 1])
    assert overrides == {
        "architectures": ["Qwen3ForSequenceClassification"],
        "classifier_from_token": ["no", "yes"],
        "is_original_qwen3_reranker": True,
    }
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert all(len(revision) == 40 for _, revision in module.MODELS.values())
    template = (ROOT / "compute/qwen3_reranker.jinja").read_text()
    assert "<Instruct>: {{ instruction" in template and "<think>\n\n</think>" in template


def test_proxy_authenticates_and_translates_to_tei_format() -> None:
    module = load_reranker()
    forwarded: list[dict[str, object]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/rerank"
        body = json.loads(request.content)
        forwarded.append(body)
        scores = [0.2, 0.8, 0.5][: len(body["documents"])]
        return httpx.Response(
            200,
            json={"results": [{"index": i, "relevance_score": s} for i, s in enumerate(scores)]},
        )

    client = httpx.AsyncClient(base_url="http://vllm", transport=httpx.MockTransport(upstream))
    api = TestClient(module.create_proxy("0.6b", "secret", client))
    body = {"query": "q", "texts": ["a", "b", "c"], "truncate": True}
    assert api.post("/rerank", json=body).status_code == 401
    assert (
        api.post("/rerank", json=body, headers={"Authorization": "Bearer nope"}).status_code == 401
    )
    response = api.post("/rerank", json=body, headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200
    assert response.json() == [
        {"index": 1, "score": 0.8},
        {"index": 2, "score": 0.5},
        {"index": 0, "score": 0.2},
    ]
    assert forwarded[0]["instruction"] == module.DEFAULT_INSTRUCTION
    assert forwarded[0]["documents"] == ["a", "b", "c"]
    assert response.headers["x-reranker-model"].startswith("Qwen/Qwen3-Reranker-0.6B@")
    assert api.get("/health").json() == {"status": "ok"}
    too_many = {"query": "q", "texts": ["x"] * 101}
    auth = {"Authorization": "Bearer secret"}
    assert api.post("/rerank", json=too_many, headers=auth).status_code == 422
    with pytest.raises(ValueError):
        module.create_proxy("0.6b", " ", client)


def test_proxy_rejects_incomplete_upstream() -> None:
    module = load_reranker()
    client = httpx.AsyncClient(
        base_url="http://vllm",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"results": [{"index": 0, "relevance_score": 0.1}]}
            )
        ),
    )
    api = TestClient(module.create_proxy("4b", "secret", client))
    response = api.post(
        "/rerank",
        json={"query": "q", "texts": ["a", "b"]},
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 502
