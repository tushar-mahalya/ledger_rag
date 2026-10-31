import asyncio
import json
from datetime import date
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import NAMESPACE_URL, uuid5

import httpx
import pytest
from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient, models

from ledger_rag.config import Settings
from ledger_rag.models import Chunk, Principal, QueryRequest
from ledger_rag.qdrant_store import QdrantRetriever, RetrievalError, _sparse


def chunk(chunk_id: str = "tenant/doc/0", **updates: Any) -> Chunk:
    values: dict[str, Any] = {
        "id": chunk_id,
        "document_id": "doc",
        "tenant_id": "tenant",
        "title": "Results",
        "text": "Revenue increased.",
        "parent_text": "Revenue increased.",
        "source": "https://example.test/report",
        "published_at": date(2024, 1, 1),
        "allowed_groups": [],
        "ordinal": 0,
        "content_hash": "hash",
    }
    values.update(updates)
    return Chunk(**values)


async def store() -> tuple[QdrantRetriever, AsyncMock, list[httpx.Request]]:
    settings = Settings(
        demo_token=SecretStr("demo-test-token"),
        retrieval_backend="qdrant",
        embedding_api_key=SecretStr("embedding-secret"),
        embedding_dimensions=32,
        embedding_model="test-embedding",
        embedding_base_url="https://embedding.test/v1/",
    )
    retriever = QdrantRetriever(settings)
    await retriever._client.close()
    client = AsyncMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={"dense": models.VectorParams(size=32, distance=models.Distance.COSINE)},
                sparse_vectors={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            ),
            metadata=dict(retriever._identity),
        )
    )
    client.retrieve.return_value = []
    client.query_points.return_value = SimpleNamespace(points=[])
    retriever._client = client
    await retriever._embedding_client.aclose()
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "test-embedding",
                "data": [{"index": i, "embedding": [1.0] * 32} for i in range(len(body["input"]))],
            },
        )

    retriever._embedding_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    return retriever, client, requests


async def test_upsert_metadata_uuid_and_tenant_scoped_stale_delete() -> None:
    retriever, client, requests = await store()
    chunks = [chunk(), chunk("tenant/doc/1", ordinal=1)]
    await retriever.upsert(chunks)
    points = client.upsert.call_args.kwargs["points"]
    assert points[0].id == str(uuid5(NAMESPACE_URL, chunks[0].id))
    assert points[0].payload["published_at"] == "2024-01-01"
    assert points[0].payload["published_day"] == date(2024, 1, 1).toordinal()
    assert points[0].payload["parent_text"] == chunks[0].parent_text
    assert points[0].vector["dense"] == [1.0] * 32
    assert points[0].vector["sparse"].indices
    delete_filter = client.delete.call_args.kwargs["points_selector"].filter
    assert delete_filter.must[0].match.value == "tenant"
    assert delete_filter.must[1].match.value == "doc"
    assert delete_filter.must_not[0].has_id == [point.id for point in points]
    assert client.upsert.call_args.kwargs["wait"] is True
    assert client.delete.call_args.kwargs["wait"] is True
    assert str(requests[0].url) == "https://embedding.test/v1/embeddings"
    assert requests[0].headers["authorization"] == "Bearer embedding-secret"
    assert json.loads(requests[0].content)["dimensions"] == 32
    await retriever.upsert(chunks)
    assert client.upsert.call_args.kwargs["points"][0].id == points[0].id
    await retriever.close()
    client.close.assert_awaited_once()
    assert retriever._embedding_client.is_closed


async def test_no_stale_delete_after_failed_embedding_or_upsert() -> None:
    retriever, client, _ = await store()
    client.upsert.side_effect = RuntimeError("upsert failed")
    with pytest.raises(RuntimeError):
        await retriever.upsert([chunk()])
    client.delete.assert_not_called()
    await retriever.close()


async def test_duplicate_and_cross_tenant_ids_rejected() -> None:
    retriever, client, requests = await store()
    with pytest.raises(RetrievalError, match="Duplicate"):
        await retriever.upsert([chunk(), chunk()])
    client.retrieve.return_value = [
        models.Record(
            id=str(uuid5(NAMESPACE_URL, chunk().id)),
            payload={"tenant_id": "other", "document_id": "doc"},
        )
    ]
    with pytest.raises(RetrievalError, match="different tenant"):
        await retriever.upsert([chunk()])
    assert not requests
    client.upsert.assert_not_called()
    await retriever.close()


async def test_collection_creation_and_readiness_reachability() -> None:
    retriever, client, _ = await store()
    client.collection_exists.return_value = False
    assert await retriever.ready()
    kwargs = client.create_collection.call_args.kwargs
    assert kwargs["vectors_config"]["dense"].size == 32
    assert kwargs["sparse_vectors_config"]["sparse"].modifier == models.Modifier.IDF
    assert kwargs["metadata"]["embedding_model"] == "test-embedding"
    client.get_collection.side_effect = RuntimeError("unreachable")
    assert not await retriever.ready()
    await retriever.close()


@pytest.mark.parametrize("mismatch", ["dimensions", "model", "identity", "sparse"])
async def test_incompatible_collection_is_rejected(mismatch: str) -> None:
    retriever, client, requests = await store()
    config = client.get_collection.return_value.config
    if mismatch == "dimensions":
        config.params.vectors["dense"].size = 64
    elif mismatch == "model":
        config.metadata["embedding_model"] = "other"
    elif mismatch == "identity":
        config.metadata = None
    else:
        config.params.sparse_vectors["sparse"].modifier = None
    with pytest.raises(RetrievalError, match="incompatible"):
        await retriever.search(
            QueryRequest(question="revenue"), Principal(subject="user", tenant_id="tenant")
        )
    assert not requests
    assert not await retriever.ready()
    await retriever.close()


async def test_hybrid_filters_before_retrieval_and_defense_in_depth() -> None:
    retriever, client, _ = await store()
    rows = [
        chunk(),
        chunk("private", allowed_groups=["finance"]),
        chunk("wrong-tenant", tenant_id="other"),
        chunk("forbidden", allowed_groups=["admin"]),
        chunk("future", published_at=date(2025, 1, 1)),
    ]
    client.query_points.return_value = SimpleNamespace(
        points=[
            models.ScoredPoint(id=i, version=1, score=1, payload=row.model_dump(mode="json"))
            for i, row in enumerate(rows)
        ]
    )
    query = QueryRequest(question="revenue", as_of=date(2024, 12, 31), top_k=5)
    principal = Principal(subject="user", tenant_id="tenant", groups=["finance"])
    hits = await retriever.search(query, principal)
    assert [hit.chunk.id for hit in hits] == ["tenant/doc/0", "private"]
    kwargs = client.query_points.call_args.kwargs
    assert kwargs["query"].fusion == models.Fusion.RRF
    query_filter = kwargs["query_filter"]
    assert query_filter.must[0].match.value == "tenant"
    assert query_filter.must[1].should[0].values_count.lte == 0
    assert query_filter.must[1].should[1].match.any == ["finance"]
    assert query.as_of is not None
    assert query_filter.must[2].range.lte == query.as_of.toordinal()
    assert all(prefetch.filter == query_filter for prefetch in kwargs["prefetch"])
    assert [prefetch.using for prefetch in kwargs["prefetch"]] == ["dense", "sparse"]
    await retriever.close()


def test_public_only_filter_and_deterministic_sparse() -> None:
    query_filter = QdrantRetriever._filter(
        QueryRequest(question="revenue"), Principal(subject="user", tenant_id="tenant")
    )
    assert isinstance(query_filter.must, list)
    acl = query_filter.must[1]
    assert isinstance(acl, models.Filter)
    assert isinstance(acl.should, list)
    assert len(acl.should) == 1
    assert _sparse("Revenue debt revenue", document=True) == _sparse(
        "REVENUE debt revenue", document=True
    )
    vector = _sparse("revenue revenue", document=False)
    assert len(vector.indices) == 1 and vector.values == [1.0]
    assert _sparse("", document=False).indices == []


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"index": 0, "embedding": [1.0]}],
        [{"index": 1, "embedding": [1.0] * 32}],
        [{"index": 0, "embedding": [0.0] * 32}],
        [{"index": 0, "embedding": [True] * 32}],
        [{"index": 0, "embedding": ["1"] * 32}],
    ],
)
async def test_invalid_embeddings_do_not_write(rows: Any) -> None:
    retriever, client, _ = await store()
    await retriever._embedding_client.aclose()
    retriever._embedding_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": rows}))
    )
    with pytest.raises(RetrievalError, match="Invalid embedding"):
        await retriever.upsert([chunk()])
    client.upsert.assert_not_called()
    client.delete.assert_not_called()
    await retriever.close()


@pytest.mark.parametrize("status", [401, 429, 500])
async def test_embedding_http_errors_are_sanitized_not_retried(status: int) -> None:
    retriever, client, _ = await store()
    await retriever._embedding_client.aclose()
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, json={"error": "embedding-secret Revenue increased."})

    retriever._embedding_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    with pytest.raises(RetrievalError) as error:
        await retriever.upsert([chunk()])
    assert str(error.value) == f"Embedding endpoint returned HTTP {status}"
    assert len(requests) == 1
    client.upsert.assert_not_called()
    client.delete.assert_not_called()
    await retriever.close()


async def test_embedding_reorders_indices_and_rejects_model_mismatch() -> None:
    retriever, _, _ = await store()
    await retriever._embedding_client.aclose()
    retriever._embedding_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [2.0] * 32},
                        {"index": 0, "embedding": [1.0] * 32},
                    ]
                },
            )
        )
    )
    assert await retriever._embed(["one", "two"]) == [[1.0] * 32, [2.0] * 32]
    await retriever._embedding_client.aclose()
    retriever._embedding_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "model": "different",
                    "data": [{"index": 0, "embedding": [1.0] * 32}],
                },
            )
        )
    )
    with pytest.raises(RetrievalError, match="Invalid embedding"):
        await retriever._embed(["one"])
    await retriever.close()


async def test_local_sdk_executes_real_sparse_dense_fusion() -> None:
    retriever, mock, _ = await store()
    await mock.close()
    retriever._client = AsyncQdrantClient(":memory:")
    assert await retriever.ready()
    await retriever.upsert(
        [
            chunk(),
            chunk("private", document_id="private", allowed_groups=["finance"]),
            chunk("foreign", document_id="foreign", tenant_id="foreign"),
        ]
    )
    hits = await retriever.search(
        QueryRequest(question="revenue"), Principal(subject="user", tenant_id="tenant")
    )
    assert [hit.chunk.id for hit in hits] == ["tenant/doc/0"]
    await retriever.upsert([chunk("replacement")])
    hits = await retriever.search(
        QueryRequest(question="revenue"), Principal(subject="user", tenant_id="tenant")
    )
    assert [hit.chunk.id for hit in hits] == ["replacement"]
    await retriever.close()


async def test_local_sdk_acl_asof_inclusive_and_tenant_scoped_replacement() -> None:
    retriever, mock, _ = await store()
    await mock.close()
    retriever._client = AsyncQdrantClient(":memory:")
    await retriever.upsert(
        [
            chunk(),
            chunk("finance", document_id="finance", allowed_groups=["finance", "audit"]),
            chunk("admin", document_id="admin", allowed_groups=["admin"]),
            chunk("future", document_id="future", published_at=date(2024, 1, 2)),
            chunk("foreign", tenant_id="foreign"),
        ]
    )
    query = QueryRequest(question="revenue", as_of=date(2024, 1, 1), top_k=10)
    public = Principal(subject="public", tenant_id="tenant")
    finance = Principal(subject="finance", tenant_id="tenant", groups=["finance"])
    foreign = Principal(subject="foreign", tenant_id="foreign")
    assert {hit.chunk.id for hit in await retriever.search(query, public)} == {"tenant/doc/0"}
    assert {hit.chunk.id for hit in await retriever.search(query, finance)} == {
        "tenant/doc/0",
        "finance",
    }
    assert {hit.chunk.id for hit in await retriever.search(query, foreign)} == {"foreign"}
    await retriever.upsert([chunk("replacement", allowed_groups=["admin"])])
    assert await retriever.search(query, public) == []
    assert {hit.chunk.id for hit in await retriever.search(query, finance)} == {"finance"}
    assert {hit.chunk.id for hit in await retriever.search(query, foreign)} == {"foreign"}
    admin = Principal(subject="admin", tenant_id="tenant", groups=["admin"])
    assert {hit.chunk.id for hit in await retriever.search(query, admin)} == {
        "replacement",
        "admin",
    }
    assert not await retriever.search(
        QueryRequest(question="revenue", as_of=date(2023, 12, 31)), finance
    )
    await retriever.close()


@pytest.mark.parametrize("operation", ["upsert", "delete"])
async def test_uncertain_mutation_quarantines_instance(operation: str) -> None:
    retriever, client, _ = await store()
    getattr(client, operation).side_effect = RuntimeError("uncertain write")
    with pytest.raises(RuntimeError, match="uncertain write"):
        await retriever.upsert([chunk()])
    assert not await retriever.ready()
    with pytest.raises(RetrievalError, match="reconciliation required"):
        await retriever.search(
            QueryRequest(question="revenue"), Principal(subject="public", tenant_id="tenant")
        )
    with pytest.raises(RetrievalError, match="reconciliation required"):
        await retriever.upsert([chunk()])
    client.query_points.assert_not_called()
    await retriever.close()


async def test_concurrent_updates_and_search_are_serialized() -> None:
    retriever, client, _ = await store()
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked_upsert(*args: Any, **kwargs: Any) -> None:
        entered.set()
        await release.wait()

    client.upsert.side_effect = blocked_upsert
    first = asyncio.create_task(retriever.upsert([chunk("first")]))
    await entered.wait()
    second = asyncio.create_task(retriever.upsert([chunk("second")]))
    search = asyncio.create_task(
        retriever.search(
            QueryRequest(question="revenue"), Principal(subject="public", tenant_id="tenant")
        )
    )
    await asyncio.sleep(0)
    assert client.upsert.await_count == 1
    client.query_points.assert_not_called()
    release.set()
    await asyncio.gather(first, second, search)
    assert client.upsert.await_count == 2
    assert client.delete.await_count == 2
    await retriever.close()


async def test_cancelled_remote_write_quarantines_instance() -> None:
    retriever, client, _ = await store()
    entered, blocked = asyncio.Event(), asyncio.Event()

    async def remote_write(*args: Any, **kwargs: Any) -> None:
        entered.set()
        await blocked.wait()

    client.upsert.side_effect = remote_write
    task = asyncio.create_task(retriever.upsert([chunk()]))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not await retriever.ready()
    with pytest.raises(RetrievalError, match="reconciliation required"):
        await retriever.search(
            QueryRequest(question="revenue"), Principal(subject="public", tenant_id="tenant")
        )
    client.delete.assert_not_called()
    await retriever.close()
