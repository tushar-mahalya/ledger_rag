"""Copy a verified Qdrant collection into an isolated quantization ablation index.

The source is read-only. The destination must not exist and is created using
QdrantRetriever's production schema/configuration checks. Existing vector values,
payloads, and IDs are copied without re-embedding or exposing document contents.
"""

import argparse
import asyncio
import os
from typing import Any, Literal

from pydantic import SecretStr
from qdrant_client import models

from ledger_rag.core.config import Settings
from ledger_rag.retrieval.qdrant import QdrantRetriever

Quantization = Literal["none", "scalar", "binary"]


def settings(
    *,
    url: str,
    collection: str,
    dimensions: int,
    embedding_url: str,
    embedding_model: str,
    quantization: Quantization,
) -> Settings:
    qdrant_key = os.environ.get("LEDGER_QDRANT_API_KEY")
    return Settings(
        demo_token=SecretStr("offline-benchmark-only"),
        retrieval_backend="qdrant",
        qdrant_url=url,
        qdrant_collection=collection,
        qdrant_api_key=SecretStr(qdrant_key) if qdrant_key else None,
        embedding_base_url=embedding_url,
        embedding_api_key=SecretStr("not-used-for-vector-copy"),
        embedding_model=embedding_model,
        embedding_dimensions=dimensions,
        qdrant_quantization=quantization,
    )


async def clone(
    *,
    url: str,
    source: str,
    destination: str,
    dimensions: int,
    embedding_url: str,
    embedding_model: str,
    quantization: Quantization,
    batch_size: int,
) -> int:
    source_store = QdrantRetriever(
        settings(
            url=url,
            collection=source,
            dimensions=dimensions,
            embedding_url=embedding_url,
            embedding_model=embedding_model,
            quantization="none",
        )
    )
    destination_store = QdrantRetriever(
        settings(
            url=url,
            collection=destination,
            dimensions=dimensions,
            embedding_url=embedding_url,
            embedding_model=embedding_model,
            quantization=quantization,
        )
    )
    try:
        if not await source_store._client.collection_exists(source):
            raise ValueError(f"Source collection does not exist: {source}")
        if await destination_store._client.collection_exists(destination):
            raise ValueError(f"Destination collection already exists: {destination}")
        await source_store._ensure_collection(create=False)
        source_count = (await source_store._client.count(collection_name=source, exact=True)).count
        await destination_store._ensure_collection(create=True)

        offset: Any | None = None
        copied = 0
        while True:
            records, next_offset = await source_store._client.scroll(
                collection_name=source,
                offset=offset,
                limit=batch_size,
                with_payload=True,
                with_vectors=True,
            )
            points = []
            for record in records:
                if record.vector is None:
                    raise RuntimeError(f"Source record has no vector: {record.id}")
                points.append(
                    models.PointStruct.model_validate(
                        {"id": record.id, "vector": record.vector, "payload": record.payload}
                    )
                )
            if points:
                await destination_store._client.upsert(
                    collection_name=destination, points=points, wait=True
                )
                copied += len(points)
                print(f"\rcopied {copied} points", end="", flush=True)
            if next_offset is None:
                break
            offset = next_offset
        print()
        destination_count = (
            await destination_store._client.count(collection_name=destination, exact=True)
        ).count
        if copied != source_count or destination_count != source_count:
            raise RuntimeError(
                f"Incomplete copy to {destination}: source={source_count}, "
                f"copied={copied}, destination={destination_count}; inspect before retrying"
            )
        await destination_store._ensure_collection(create=False)
        return copied
    finally:
        await source_store.close()
        await destination_store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--embedding-model", default="Alibaba-NLP/gte-modernbert-base")
    parser.add_argument("--quantization", choices=["scalar", "binary"], required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.source == args.destination:
        parser.error("--destination must differ from --source")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    copied = asyncio.run(
        clone(
            url=args.qdrant_url,
            source=args.source,
            destination=args.destination,
            dimensions=args.dimensions,
            embedding_url=args.embedding_url,
            embedding_model=args.embedding_model,
            quantization=args.quantization,
            batch_size=args.batch_size,
        )
    )
    print(f"copied {copied} points from {args.source} to {args.destination}")


if __name__ == "__main__":
    main()
