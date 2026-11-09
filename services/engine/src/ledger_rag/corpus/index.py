"""Versioned builds behind a serving alias: list, promote, roll back, prune.

Each build of a corpus is an immutable collection ``<alias>__<build_id>`` carrying its
own build record (status and lineage) in the collection metadata. Serving reads the
alias. Promotion is one atomic alias swap, so readers never see a half-built index,
and rollback is the same swap back to the previous verified build.
"""

from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from ledger_rag.retrieval.qdrant import BUILD_RECORD_KEY

PROMOTABLE = {"verified", "superseded", "promoted"}


class PromotionError(RuntimeError):
    """The requested promotion would expose an unverified or foreign index."""


@dataclass(frozen=True)
class Build:
    collection: str
    build_id: str
    status: str
    record: dict[str, Any]


class IndexRegistry:
    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client

    async def current(self, alias: str) -> str | None:
        response = await self._client.get_aliases()
        for item in response.aliases:
            if item.alias_name == alias:
                return item.collection_name
        return None

    async def record(self, collection: str) -> dict[str, Any] | None:
        info = await self._client.get_collection(collection)
        record = (info.config.metadata or {}).get(BUILD_RECORD_KEY)
        return record if isinstance(record, dict) else None

    async def builds(self, alias: str) -> list[Build]:
        """Builds of ``alias``, oldest first (build ids sort by creation time)."""
        prefix = f"{alias}__"
        response = await self._client.get_collections()
        names = sorted(c.name for c in response.collections if c.name.startswith(prefix))
        builds = []
        for name in names:
            record = await self.record(name) or {}
            builds.append(
                Build(name, name[len(prefix) :], str(record.get("status", "unknown")), record)
            )
        return builds

    async def _set_status(self, collection: str, status: str) -> None:
        record = await self.record(collection)
        if record is not None:
            await self._client.update_collection(
                collection, metadata={BUILD_RECORD_KEY: {**record, "status": status}}
            )

    async def promote(self, alias: str, collection: str, *, adopt: bool = False) -> str | None:
        """Point ``alias`` at ``collection`` atomically; return the previous target.

        Only verified builds are promotable. ``adopt`` admits a collection built before
        build records existed; the engine still checks its embedding and chunk identity
        when it starts serving it.
        """
        if not await self._client.collection_exists(collection):
            raise PromotionError(f"Collection {collection!r} does not exist")
        record = await self.record(collection)
        if record is None and not adopt:
            raise PromotionError(
                f"{collection!r} has no build record; use --adopt for legacy indexes"
            )
        if record is not None and record.get("status") not in PROMOTABLE:
            raise PromotionError(f"{collection!r} is {record.get('status')!r}, not verified")
        previous = await self.current(alias)
        operations: list[models.AliasOperations] = []
        if previous is not None:
            operations.append(
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)
            )
        )
        await self._client.update_collection_aliases(change_aliases_operations=operations)
        if previous is not None and previous != collection:
            await self._set_status(previous, "superseded")
        await self._set_status(collection, "promoted")
        return previous

    async def rollback(self, alias: str) -> str:
        """Re-point ``alias`` at the newest earlier promotable build; return it."""
        current = await self.current(alias)
        candidates = [
            build
            for build in await self.builds(alias)
            if build.collection != current and build.status in PROMOTABLE
        ]
        if current is not None:
            candidates = [b for b in candidates if b.collection < current]
        if not candidates:
            raise PromotionError(f"No earlier verified build of {alias!r} to roll back to")
        target = candidates[-1].collection
        await self.promote(alias, target)
        return target

    async def prune(self, alias: str, keep: int) -> list[str]:
        """Delete builds beyond the newest ``keep`` that are not serving. Irreversible."""
        current = await self.current(alias)
        builds = [b for b in await self.builds(alias) if b.collection != current]
        doomed = builds[: max(0, len(builds) - max(0, keep - 1))]
        for build in doomed:
            await self._client.delete_collection(build.collection)
        return [build.collection for build in doomed]
