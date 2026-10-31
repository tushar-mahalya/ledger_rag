"""Remote dense/BM25 hybrid retrieval, isolated by tenant and ACL.

Sparse TF uses BM25 k1=1.2, b=0.75 and a fixed reference length of 256
tokens; Qdrant supplies corpus IDF. No local embedding model is loaded.
Upsert receives full document chunk sets. Stale deletion follows successful
upsert, is tenant-scoped, and is not transactional with it. Within an
instance, reads run concurrently while writes and collection management are
exclusive (writer-preferring), so a search never observes this instance's own
half-applied replacement. The instance refuses further operations after an
uncertain mutation; recovery requires external reconciliation, not a retry.

Write uncertainty is also durable and shared: before mutating, a writer stores
a lease ``{writer, expires_at}`` in the collection metadata and removes it only
after every upsert and stale delete succeeded. Every instance already reads the
collection metadata on each search and readiness check, so an *expired* lease
(writer crashed, was cancelled or restarted mid-replacement) fails closed in all
instances and across restarts, at no extra round trip. An unexpired foreign lease
is an in-flight write: reads continue (replacement was never transactional) and
other writers refuse. The check-then-set is best effort, not a distributed lock;
concurrent writers still need external serialization. Recovery is explicit:
reconcile, then ``release_write_lease()``.

Readiness is read-only: it validates reachability, schema/identity and write
state and never creates or alters a collection. Provisioning (collection,
sparse IDF modifier, metadata and payload indexes) happens only on explicit
paths: ``bootstrap()`` (``LEDGER_QDRANT_BOOTSTRAP_ON_STARTUP``), document
upsert/ingestion, or ``create_empty_collection()`` for staged builds.
"""

import asyncio
import hashlib
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

import httpx
import structlog
from prometheus_client import Counter as PrometheusCounter
from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient, models

from ledger_rag.core.config import Settings
from ledger_rag.core.models import Chunk, Hit, Principal, QueryRequest
from ledger_rag.generation.providers import _endpoint
from ledger_rag.ingestion.chunking import contextual_text


class RetrievalError(RuntimeError):
    """A retrieval transport, embedding or collection compatibility failure."""


WRITE_LEASE_KEY = "ledger_write_lease"
logger = structlog.get_logger(__name__)


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


class _ReadWriteGate:
    """Writer-preferring async read/write gate; state changes never await, so it is cancel-safe."""

    def __init__(self) -> None:
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0
        self._waiters: list[asyncio.Future[None]] = []

    def _wake(self) -> None:
        waiters, self._waiters = self._waiters, []
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(None)

    async def _wait_until(self, ready: Callable[[], bool]) -> None:
        while not ready():
            waiter = asyncio.get_running_loop().create_future()
            self._waiters.append(waiter)
            await waiter

    @asynccontextmanager
    async def read(self) -> AsyncIterator[None]:
        await self._wait_until(lambda: not self._writer and not self._waiting_writers)
        self._readers += 1
        try:
            yield
        finally:
            self._readers -= 1
            self._wake()

    @asynccontextmanager
    async def write(self) -> AsyncIterator[None]:
        self._waiting_writers += 1
        try:
            await self._wait_until(lambda: not self._writer and not self._readers)
        finally:
            self._waiting_writers -= 1
            self._wake()
        self._writer = True
        try:
            yield
        finally:
            self._writer = False
            self._wake()


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
        self._gate = _ReadWriteGate()
        self._write_uncertain = False
        self._writer_id = uuid4().hex
        self._lease_expires = 0.0
        self._staging_collection = False
        self._rejected_points = PrometheusCounter(
            "ledger_qdrant_rejected_points",
            "Retrieved Qdrant points dropped as corrupt or filter-violating",
            ["reason"],
            registry=None,
        )
        self._identity = {
            "ledger_schema": 2,
            "chunk_text": "title-header-v1",
            "embedding_model": settings.embedding_model,
            "embedding_endpoint": settings.embedding_base_url.rstrip("/"),
            "embedding_dimensions": settings.embedding_dimensions,
            "sparse_model": "hashed-bm25-k1=1.2-b=0.75-avgdl=256-blake2s32-v1",
        }
        if settings.embedding_space_id:
            self._identity["ledger_schema"] = 3
            self._identity["embedding_space_id"] = settings.embedding_space_id
            del self._identity["embedding_endpoint"]

    async def _create_collection(self) -> None:
        created = await self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=self._settings.embedding_dimensions,
                    distance=models.Distance.COSINE,
                    on_disk=True if self._settings.qdrant_quantization != "none" else None,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
            quantization_config=self._quantization(),
            metadata=self._identity,
        )
        if not created:
            raise RetrievalError("Qdrant did not acknowledge collection creation")

    async def create_empty_collection(self) -> None:
        """Claim a new staging collection; never reuse an existing or partial index."""
        async with self._gate.write():
            self._check_write_state()
            if await self._client.collection_exists(self._collection):
                raise RetrievalError("Destination collection already exists; use a new name")
            await self._create_collection()
            await self._ensure_collection(create=False)
            await self._create_payload_indexes()
            self._staging_collection = True

    async def _create_payload_indexes(self) -> None:
        indexes: dict[str, models.PayloadSchemaType | models.KeywordIndexParams] = {
            "tenant_id": models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD, is_tenant=True
            ),
            "document_id": models.PayloadSchemaType.KEYWORD,
            "allowed_groups": models.PayloadSchemaType.KEYWORD,
            "published_day": models.PayloadSchemaType.INTEGER,
        }
        for name, schema in indexes.items():
            await self._client.create_payload_index(
                self._collection, field_name=name, field_schema=schema, wait=True
            )

    async def bootstrap(self) -> None:
        """Explicitly provision a missing collection with indexes, then validate it.

        An existing collection is validated but never altered.
        """
        async with self._gate.write():
            self._check_write_state()
            await self._ensure_collection(create=True)

    async def count(self) -> int:
        async with self._gate.read():
            self._check_write_state()
            await self._ensure_collection(create=False)
            return (await self._client.count(self._collection, exact=True)).count

    async def _ensure_collection(self, *, create: bool) -> dict[str, Any]:
        """Validate the collection; ``create`` is only legal while holding the write gate."""
        if create and not await self._client.collection_exists(self._collection):
            try:
                await self._create_collection()
            except Exception:
                # Another process may have provisioned it concurrently; validate theirs.
                if not await self._client.collection_exists(self._collection):
                    raise
            else:
                await self._create_payload_indexes()
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
        if self._settings.qdrant_quantization != "none":
            quantization = vectors["dense"].quantization_config
            if quantization is None:
                quantization = info.config.quantization_config
            if quantization != self._quantization():
                raise RetrievalError(
                    "Qdrant quantization configuration is incompatible; "
                    "provision or migrate the collection explicitly"
                )
        lease = metadata.get(WRITE_LEASE_KEY)
        if lease is not None and not self._owns(lease) and not self._lease_active(lease):
            raise RetrievalError(
                "Qdrant collection has an abandoned write lease; reconciliation required"
            )
        return metadata

    def _owns(self, lease: object) -> bool:
        return isinstance(lease, dict) and lease.get("writer") == self._writer_id

    @staticmethod
    def _lease_active(lease: object) -> bool:
        expires = lease.get("expires_at") if isinstance(lease, dict) else None
        return (
            isinstance(expires, int | float)
            and not isinstance(expires, bool)
            and math.isfinite(expires)
            and expires > time.time()
        )

    async def _set_lease(self) -> None:
        expires = time.time() + self._settings.qdrant_write_lease_seconds
        await self._client.update_collection(
            self._collection,
            metadata={WRITE_LEASE_KEY: {"writer": self._writer_id, "expires_at": expires}},
        )
        self._lease_expires = expires

    async def release_write_lease(self) -> None:
        """Operator acknowledgement that the collection was reconciled after a failed write."""
        async with self._gate.write():
            await self._client.update_collection(self._collection, metadata={WRITE_LEASE_KEY: None})
            self._write_uncertain = False

    def _quantization(self) -> models.QuantizationConfig | None:
        mode = self._settings.qdrant_quantization
        if mode == "scalar":
            return models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, quantile=0.99, always_ram=True
                )
            )
        if mode == "binary":
            return models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True)
            )
        return None

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        size = self._settings.embedding_batch_size
        vectors: list[list[float]] = []
        for start in range(0, len(texts), size):
            vectors.extend(await self._embed_batch(texts[start : start + size]))
        return vectors

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
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
        async with self._gate.write():
            self._check_write_state()
            await self._upsert(chunks)

    async def append_staging(self, chunks: list[Chunk]) -> None:
        """Skip redundant stale deletes only for a collection claimed by this instance."""
        async with self._gate.write():
            self._check_write_state()
            if not self._staging_collection:
                raise RetrievalError("Staging writes require a newly claimed collection")
            await self._upsert(chunks, replace_documents=False)

    def _check_write_state(self) -> None:
        if self._write_uncertain:
            raise RetrievalError("Qdrant mutation outcome is uncertain; reconciliation required")

    async def _upsert(self, chunks: list[Chunk], *, replace_documents: bool = True) -> None:
        if not chunks:
            return
        if len({chunk.id for chunk in chunks}) != len(chunks):
            raise RetrievalError("Duplicate chunk IDs in upsert")
        lease = (await self._ensure_collection(create=True)).get(WRITE_LEASE_KEY)
        if lease is not None and not self._owns(lease):
            raise RetrievalError("Another writer holds the Qdrant write lease")
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
        embeddings = await self._embed([contextual_text(chunk) for chunk in chunks])
        points = []
        documents: dict[tuple[str, str], list[str]] = defaultdict(list)
        for chunk, point_id, dense in zip(chunks, ids, embeddings, strict=True):
            payload = chunk.model_dump(mode="json")
            payload["published_day"] = chunk.published_at.toordinal()
            points.append(
                models.PointStruct(
                    id=point_id,
                    payload=payload,
                    vector={
                        "dense": dense,
                        "sparse": _sparse(contextual_text(chunk), document=True),
                    },
                )
            )
            documents[(chunk.tenant_id, chunk.document_id)].append(point_id)
        await self._set_lease()
        try:
            await self._apply(points, documents, replace_documents=replace_documents)
        except BaseException:
            # Failure or cancellation mid-mutation leaves an unknown mixture of chunks;
            # the durable lease is deliberately left to expire so every instance fails closed.
            self._write_uncertain = True
            raise
        await self._client.update_collection(self._collection, metadata={WRITE_LEASE_KEY: None})

    async def _apply(
        self,
        points: list[models.PointStruct],
        documents: dict[tuple[str, str], list[str]],
        *,
        replace_documents: bool,
    ) -> None:
        size = self._settings.qdrant_upsert_batch_size
        for start in range(0, len(points), size):
            if self._lease_expires - time.time() < self._settings.qdrant_write_lease_seconds / 2:
                await self._set_lease()
            await self._client.upsert(
                self._collection, points=points[start : start + size], wait=True
            )
        if replace_documents:
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
        async with self._gate.read():
            self._check_write_state()
            return await self._search(query, principal)

    async def _search(self, query: QueryRequest, principal: Principal) -> list[Hit]:
        await self._ensure_collection(create=False)
        fusion = self._settings.retrieval_fusion
        query_filter = self._filter(query, principal)
        limit = max(query.top_k * 4, 20)
        prefetch: list[models.Prefetch] = []
        if fusion in {"hybrid", "dense"}:
            dense = (await self._embed([self._settings.embedding_query_prefix + query.question]))[0]
            params = models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=self._settings.qdrant_quantization == "none",
                    rescore=True,
                    oversampling=self._settings.qdrant_oversampling,
                )
            )
            prefetch.append(
                models.Prefetch(
                    query=dense, using="dense", filter=query_filter, limit=limit, params=params
                )
            )
        if fusion in {"hybrid", "sparse"}:
            prefetch.append(
                models.Prefetch(
                    query=_sparse(query.question, document=False),
                    using="sparse",
                    filter=query_filter,
                    limit=limit,
                )
            )
        hits: list[Hit] = []
        rejected: Counter[str] = Counter()
        offset = 0
        # Refill past rejected points from the remaining fused candidates (bounded by the
        # prefetch pool); the common, clean case still costs a single top_k request.
        while len(hits) < query.top_k and offset < limit:
            page = min(query.top_k - len(hits), limit - offset)
            result = await self._client.query_points(
                self._collection,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                limit=page,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            offset += len(result.points)
            for point in result.points:
                try:
                    chunk = Chunk.model_validate(point.payload)
                except ValidationError:
                    rejected["invalid_payload"] += 1
                    continue
                if (
                    chunk.tenant_id != principal.tenant_id
                    or (
                        chunk.allowed_groups
                        and not set(chunk.allowed_groups) & set(principal.groups)
                    )
                    or (query.as_of and chunk.published_at > query.as_of)
                ):
                    rejected["filter_violation"] += 1
                    continue
                if not math.isfinite(point.score):
                    rejected["invalid_score"] += 1
                    continue
                hits.append(Hit(chunk=chunk, score=point.score))
            if len(result.points) < page:
                break
        if rejected:
            for reason, count in rejected.items():
                self._rejected_points.labels(reason).inc(count)
            # Counts only: payloads and point IDs may identify restricted documents.
            logger.warning(
                "qdrant_points_rejected",
                collection=self._collection,
                reasons=dict(rejected),
                returned=len(hits),
                requested=query.top_k,
            )
        return hits[: query.top_k]

    def collectors(self) -> list[PrometheusCounter]:
        """Metrics owned by this retriever, for registration in the app registry."""
        return [self._rejected_points]

    async def ready(self) -> bool:
        """Read-only and lock-free: never creates, alters or waits for writers."""
        try:
            self._check_write_state()
            await self._ensure_collection(create=False)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        try:
            await self._embedding_client.aclose()
        finally:
            await self._client.close()
