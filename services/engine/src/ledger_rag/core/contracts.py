from typing import Any, Protocol

from ledger_rag.core.models import Chunk, Hit, Principal, QueryRequest


class Retriever(Protocol):
    async def upsert(self, chunks: list[Chunk]) -> None: ...

    async def search(self, query: QueryRequest, principal: Principal) -> list[Hit]: ...

    async def update_document_acl(
        self, tenant_id: str, document_id: str, allowed_groups: list[str]
    ) -> int:
        """Replace one document's ACL on every chunk (no re-embedding); chunks changed."""
        ...

    async def delete_document(self, tenant_id: str, document_id: str) -> int:
        """Remove every chunk of one tenant's document; chunks deleted."""
        ...

    async def ready(self) -> bool: ...

    async def close(self) -> None: ...


class Provider(Protocol):
    model: str

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Return OpenAI-normalized {content, tool_calls} for either wire protocol."""
        ...

    async def close(self) -> None: ...
