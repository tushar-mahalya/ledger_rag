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

from ledger_rag.core.config import Settings
from ledger_rag.core.models import Chunk, Hit, Principal, QueryRequest
from ledger_rag.generation.providers import _endpoint
from ledger_rag.ingestion.chunking import contextual_text


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


# Text formula each chunk profile embeds; v1 keeps its original identity value.
CHUNK_TEXT = {"v1": "title-header-v1", "v2": "contextual-v2", "v3": "contextual-v3"}


def _describe_profile(profile: object, summary_version: object) -> str:
    return f"{profile} (summaries {summary_version})" if summary_version else str(profile)


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
        self._staging_collection = False
        self._identity: dict[str, str | int] = {
            "ledger_schema": 2,
            "chunk_text": CHUNK_TEXT[settings.chunk_profile],
            "embedding_model": settings.embedding_model,
            "embedding_endpoint": settings.embedding_base_url.rstrip("/"),
            "embedding_dimensions": settings.embedding_dimensions,
            "sparse_model": "hashed-bm25-k1=1.2-b=0.75-avgdl=256-blake2s32-v1",
        }
        if settings.embedding_space_id:
            self._identity["ledger_schema"] = 3
            self._identity["embedding_space_id"] = settings.embedding_space_id
            del self._identity["embedding_endpoint"]
        if settings.chunk_profile != "v1":
            # v1 collections predate these keys, so their absence means v1.
            self._identity["chunk_profile"] = settings.chunk_profile
        if settings.chunk_summary_version:
            self._identity["chunk_summary_version"] = settings.chunk_summary_version

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
        async with self._operation_lock:
            self._check_write_state()
            async with self._lock:
                if await self._client.collection_exists(self._collection):
                    raise RetrievalError("Destination collection already exists; use a new name")
                await self._create_collection()
            await self._ensure_collection(create=False)
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
            self._staging_collection = True

    async def count(self) -> int:
        async with self._operation_lock:
            self._check_write_state()
            await self._ensure_collection(create=False)
            return (await self._client.count(self._collection, exact=True)).count

    def _check_chunk_identity(self, metadata: dict[str, object]) -> None:
        stored = (
            metadata.get("chunk_profile", "v1"),
            metadata.get("chunk_summary_version", ""),
        )
        configured = (self._settings.chunk_profile, self._settings.chunk_summary_version)
        if stored != configured:
            raise RetrievalError(
                f"Qdrant collection {self._collection!r} was built with chunk profile "
                f"{_describe_profile(*stored)} but the service is configured for "
                f"{_describe_profile(*configured)}; set LEDGER_CHUNK_PROFILE and "
                "LEDGER_CHUNK_SUMMARY_VERSION to match the collection or point "
                "LEDGER_QDRANT_COLLECTION at a matching collection"
            )

    async def _ensure_collection(self, *, create: bool) -> None:
        async with self._lock:
            if create and not await self._client.collection_exists(self._collection):
                await self._create_collection()
            info = await self._client.get_collection(self._collection)
            vectors = info.config.params.vectors
            sparse = info.config.params.sparse_vectors
            metadata = info.config.metadata
            self._check_chunk_identity(metadata or {})
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
        async with self._operation_lock:
            self._check_write_state()
            await self._upsert(chunks)

    async def append_staging(self, chunks: list[Chunk]) -> None:
        """Skip redundant stale deletes only for a collection claimed by this instance."""
        async with self._operation_lock:
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
        self._write_uncertain = True
        size = self._settings.qdrant_upsert_batch_size
        for start in range(0, len(points), size):
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
