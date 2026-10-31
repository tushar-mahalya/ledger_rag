"""Remote dense/BM25 hybrid retrieval, isolated by tenant and ACL.

Sparse TF uses BM25 k1=1.2, b=0.75 and a fixed reference length of 256
tokens; Qdrant supplies corpus IDF. No local embedding model is loaded.
Upsert receives full document chunk sets. Stale deletion follows successful
upsert, is tenant-scoped, and is not transactional with it. This instance
serializes reads and writes and refuses further operations after an uncertain
mutation; recovery requires external reconciliation, not merely a retry.
Other instances require external serialization and coordinated recovery.
"""

import asyncio
import hashlib
import math
import re
from collections import Counter, defaultdict
from uuid import NAMESPACE_URL, uuid5

import httpx
from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient, models

from ledger_rag.config import Settings
from ledger_rag.models import Chunk, Hit, Principal, QueryRequest
from ledger_rag.providers import _endpoint


class RetrievalError(RuntimeError):
    """A retrieval transport, embedding or collection compatibility failure."""


def _sparse(text: str, *, document: bool) -> models.SparseVector:
    tokens = re.findall(r"\w+", text.lower())
    counts = Counter(tokens)
    weights: dict[int, float] = defaultdict(float)
    for term, frequency in counts.items():
        index = int.from_bytes(hashlib.blake2s(term.encode(), digest_size=4).digest(), "big")
        weight = (
            frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * len(tokens) / 256))
            if document
            else 1.0
        )
        weights[index] += weight
    indices = sorted(weights)
    return models.SparseVector(indices=indices, values=[weights[index] for index in indices])


class QdrantRetriever:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._collection = settings.qdrant_collection
        self._client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
            timeout=math.ceil(min(settings.query_deadline_seconds, 120)),
        )
        self._embedding_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=5),
            follow_redirects=False,
        )
        self._lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()
        self._write_uncertain = False
        self._identity = {
            "ledger_schema": 1,
            "embedding_model": settings.embedding_model,
            "embedding_endpoint": settings.embedding_base_url.rstrip("/"),
            "embedding_dimensions": settings.embedding_dimensions,
            "sparse_model": "hashed-bm25-k1=1.2-b=0.75-avgdl=256-blake2s32-v1",
        }

    async def _ensure_collection(self, *, create: bool) -> None:
        async with self._lock:
            if create and not await self._client.collection_exists(self._collection):
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=self._settings.embedding_dimensions,
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
                    },
                    metadata=self._identity,
                )
            info = await self._client.get_collection(self._collection)
            vectors = info.config.params.vectors
            sparse = info.config.params.sparse_vectors
            metadata = info.config.metadata
            if (
                not isinstance(vectors, dict)
                or "dense" not in vectors
                or vectors["dense"].size != self._settings.embedding_dimensions
                or vectors["dense"].distance != models.Distance.COSINE
                or not sparse
                or "sparse" not in sparse
                or sparse["sparse"].modifier != models.Modifier.IDF
                or not metadata
                or any(metadata.get(key) != value for key, value in self._identity.items())
            ):
                raise RetrievalError("Qdrant collection embedding/schema identity is incompatible")

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._embedding_client.post(
                _endpoint(self._settings.embedding_base_url, "embeddings"),
                headers={
                    "Authorization": "Bearer "
                    + (
                        self._settings.embedding_api_key.get_secret_value()
                        if self._settings.embedding_api_key
                        else ""
                    )
                },
                json={
                    "model": self._settings.embedding_model,
                    "input": texts,
                    "dimensions": self._settings.embedding_dimensions,
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RetrievalError(
                f"Embedding endpoint returned HTTP {exc.response.status_code}"
            ) from None
        except httpx.HTTPError:
            raise RetrievalError("Embedding endpoint transport failure") from None
        try:
            data = response.json()
            if data.get("model", self._settings.embedding_model) != self._settings.embedding_model:
                raise ValueError("Wrong embedding model")
            rows = data["data"]
            if not isinstance(rows, list) or len(rows) != len(texts):
                raise ValueError("Wrong embedding count")
            vectors: dict[int, list[float]] = {}
            for row in rows:
                index, vector = row["index"], row["embedding"]
                if (
                    type(index) is not int
                    or index not in range(len(texts))
                    or index in vectors
                    or not isinstance(vector, list)
                    or len(vector) != self._settings.embedding_dimensions
                    or any(
                        type(value) not in {int, float} or not math.isfinite(value)
                        for value in vector
                    )
                    or not any(vector)
                ):
                    raise ValueError("Invalid embedding vector")
                vectors[index] = [float(value) for value in vector]
            return [vectors[index] for index in range(len(texts))]
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
            raise RetrievalError("Invalid embedding response") from None

    async def upsert(self, chunks: list[Chunk]) -> None:
        async with self._operation_lock:
            self._check_write_state()
            await self._upsert(chunks)

    def _check_write_state(self) -> None:
        if self._write_uncertain:
            raise RetrievalError("Qdrant mutation outcome is uncertain; reconciliation required")

    async def _upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        if len({chunk.id for chunk in chunks}) != len(chunks):
            raise RetrievalError("Duplicate chunk IDs in upsert")
        await self._ensure_collection(create=True)
        ids = [str(uuid5(NAMESPACE_URL, chunk.id)) for chunk in chunks]
        owners = {point_id: chunk for point_id, chunk in zip(ids, chunks, strict=True)}
        existing = await self._client.retrieve(
            self._collection, ids=ids, with_payload=True, with_vectors=False
        )
        for point in existing:
            owner = owners[str(point.id)]
            if (
                not point.payload
                or point.payload.get("tenant_id") != owner.tenant_id
                or point.payload.get("document_id") != owner.document_id
            ):
                raise RetrievalError("Chunk ID is owned by a different tenant or document")
        embeddings = await self._embed([chunk.text for chunk in chunks])
        points = []
        documents: dict[tuple[str, str], list[str]] = defaultdict(list)
        for chunk, point_id, dense in zip(chunks, ids, embeddings, strict=True):
            payload = chunk.model_dump(mode="json")
            payload["published_day"] = chunk.published_at.toordinal()
            points.append(
                models.PointStruct(
                    id=point_id,
                    payload=payload,
                    vector={"dense": dense, "sparse": _sparse(chunk.text, document=True)},
                )
            )
            documents[(chunk.tenant_id, chunk.document_id)].append(point_id)
        self._write_uncertain = True
        await self._client.upsert(self._collection, points=points, wait=True)
        for (tenant, document), current_ids in documents.items():
            await self._client.delete(
                self._collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="tenant_id", match=models.MatchValue(value=tenant)
                            ),
                            models.FieldCondition(
                                key="document_id", match=models.MatchValue(value=document)
                            ),
                        ],
                        must_not=[
                            models.HasIdCondition(has_id=[point_id for point_id in current_ids])
                        ],
                    )
                ),
                wait=True,
            )
        self._write_uncertain = False

    @staticmethod
    def _filter(query: QueryRequest, principal: Principal) -> models.Filter:
        acl: list[models.Condition] = [
            models.FieldCondition(key="allowed_groups", values_count=models.ValuesCount(lte=0))
        ]
        if principal.groups:
            acl.append(
                models.FieldCondition(
                    key="allowed_groups", match=models.MatchAny(any=principal.groups)
                )
            )
        must: list[models.Condition] = [
            models.FieldCondition(
                key="tenant_id", match=models.MatchValue(value=principal.tenant_id)
            ),
            models.Filter(should=acl),
        ]
        if query.as_of:
            must.append(
                models.FieldCondition(
                    key="published_day", range=models.Range(lte=query.as_of.toordinal())
                )
            )
        return models.Filter(must=must)

    async def search(self, query: QueryRequest, principal: Principal) -> list[Hit]:
        async with self._operation_lock:
            self._check_write_state()
            return await self._search(query, principal)

    async def _search(self, query: QueryRequest, principal: Principal) -> list[Hit]:
        await self._ensure_collection(create=False)
        dense = (await self._embed([query.question]))[0]
        query_filter = self._filter(query, principal)
        limit = max(query.top_k * 4, 20)
        prefetch = [
            models.Prefetch(query=dense, using="dense", filter=query_filter, limit=limit),
            models.Prefetch(
                query=_sparse(query.question, document=False),
                using="sparse",
                filter=query_filter,
                limit=limit,
            ),
        ]
        result = await self._client.query_points(
            self._collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=query.top_k,
            with_payload=True,
            with_vectors=False,
        )
        hits = []
        for point in result.points:
            try:
                chunk = Chunk.model_validate(point.payload)
            except ValidationError:
                continue
            if (
                chunk.tenant_id != principal.tenant_id
                or (chunk.allowed_groups and not set(chunk.allowed_groups) & set(principal.groups))
                or (query.as_of and chunk.published_at > query.as_of)
                or not math.isfinite(point.score)
            ):
                continue
            hits.append(Hit(chunk=chunk, score=point.score))
        return hits[: query.top_k]

    async def ready(self) -> bool:
        try:
            async with self._operation_lock:
                self._check_write_state()
                await self._ensure_collection(create=True)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        try:
            await self._embedding_client.aclose()
        finally:
            await self._client.close()
