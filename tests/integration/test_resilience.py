import asyncio
import time
from datetime import date
from types import SimpleNamespace
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from fastapi.testclient import TestClient

from ledger_rag.api.app import create_app
from ledger_rag.core.config import Settings
from ledger_rag.core.models import Document, Hit, Principal, QueryRequest
from ledger_rag.ingestion.chunking import chunk_document
from ledger_rag.orchestration.service import RagService
from ledger_rag.retrieval.memory import MemoryRetriever
from ledger_rag.security.auth import Authenticator, RateLimiter
from tests.unit.test_core import FakeProvider


class FaultyRetriever(MemoryRetriever):
    async def search(self, query: QueryRequest, principal: Principal) -> list[Hit]:
        return [Hit(chunk=next(iter(self.chunks.values())), score=1)]


async def test_defense_in_depth_rejects_faulty_retriever(
    settings: Settings, document: Document, principal: Principal
) -> None:
    store = FaultyRetriever()
    await store.upsert(chunk_document(document.model_copy(update={"tenant_id": "other"})))
    provider = FakeProvider([])
    with pytest.raises(RuntimeError, match="unauthorized"):
        await RagService(settings, store, provider).retrieve(
            QueryRequest(question="Revenue"), principal
        )
    assert not provider.messages


async def test_corrective_search_cannot_escape_acl_or_time(
    settings: Settings, document: Document, principal: Principal
) -> None:
    store = MemoryRetriever()
    future = document.model_copy(
        update={"id": "future", "text": "Future liquidity 999", "published_at": date(2027, 1, 1)}
    )
    allowed = document.model_copy(
        update={"id": "cash", "text": "Cash liquidity was USD 30 million."}
    )
    private = document.model_copy(
        update={"id": "private", "text": "Private liquidity 999", "allowed_groups": ["private"]}
    )
    await store.upsert([c for d in [document, future, allowed, private] for c in chunk_document(d)])
    provider = FakeProvider(
        [
            {
                "tool_calls": [
                    {"id": "t1", "name": "search_evidence", "arguments": {"question": "liquidity"}}
                ]
            },
            {"content": "Cash was USD 30 million. [C2]"},
        ]
    )
    service = RagService(settings, store, provider)
    query = QueryRequest(question="Revenue", as_of=date(2025, 12, 31), top_k=1)
    hits, elapsed = await service.retrieve(query, principal)
    answer = await service.generate(query, hits, elapsed, "test", principal)
    assert not answer.abstained
    assert [c.document_id for c in answer.citations] == ["cash"]
    tool_result = provider.messages[1][-1]["content"]
    assert "999" not in tool_result


class FailingProvider:
    model = "failure-test"

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        raise httpx.ConnectError("Contains sensitive URL that must not be exposed")

    async def close(self) -> None:
        pass


def test_upstream_failure_is_not_success(settings: Settings, document: Document) -> None:
    app = create_app(settings, MemoryRetriever(), FailingProvider())
    auth = {"Authorization": "Bearer test-only-token"}
    with TestClient(app) as client:
        client.post(
            "/v1/documents", headers=auth, json={"documents": [document.model_dump(mode="json")]}
        )
        response = client.post("/v1/query", headers=auth, json={"question": "Revenue"})
        assert response.status_code == 502
        assert "sensitive" not in response.text
        stream = client.post("/v1/query/stream", headers=auth, json={"question": "Revenue"})
        assert "event: error" in stream.text and "event: done" not in stream.text
        assert app.state.runtime.slots._value == settings.max_concurrent_queries


class SlowProvider(FailingProvider):
    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        await asyncio.sleep(0.1)
        return {"content": "Delayed [C1]"}


def test_deadline_releases_capacity(settings: Settings, document: Document) -> None:
    app = create_app(
        settings.model_copy(update={"query_deadline_seconds": 0.01}),
        MemoryRetriever(),
        SlowProvider(),
    )
    auth = {"Authorization": "Bearer test-only-token"}
    with TestClient(app) as client:
        client.post(
            "/v1/documents", headers=auth, json={"documents": [document.model_dump(mode="json")]}
        )
        response = client.post("/v1/query", headers=auth, json={"question": "Revenue"})
        assert response.status_code == 504
        assert app.state.runtime.slots._value == settings.max_concurrent_queries


async def test_oidc_signature_and_typed_claims(settings: Settings) -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    oidc = settings.model_copy(
        update={
            "auth_mode": "oidc",
            "oidc_issuer": "https://id.example",
            "oidc_audience": "ledger",
            "oidc_jwks_url": "https://id.example/jwks",
        }
    )
    auth = Authenticator(oidc)
    auth.jwks = SimpleNamespace(
        get_signing_key_from_jwt=lambda token: SimpleNamespace(key=private_key.public_key())
    )
    claims = {
        "sub": "user",
        "tenant_id": "tenant-a",
        "roles": ["reader"],
        "groups": ["analyst"],
        "iss": "https://id.example",
        "aud": "ledger",
        "exp": int(time.time()) + 60,
        "iat": int(time.time()),
    }
    token = jwt.encode(claims, private_key, algorithm="RS256")
    result = await auth.authenticate(f"Bearer {token}")
    assert result.tenant_id == "tenant-a" and result.groups == ["analyst"]
    for changes in [
        {"aud": "wrong"},
        {"exp": 1},
        {"tenant_id": ""},
        {"groups": "admin"},
        {"roles": [42]},
        {"sub": ""},
        {"iss": "https://evil.example"},
    ]:
        invalid = jwt.encode({**claims, **changes}, private_key, algorithm="RS256")
        with pytest.raises(HTTPException) as error:
            await auth.authenticate(f"Bearer {invalid}")
        assert error.value.status_code == 401
    with pytest.raises(HTTPException):
        await auth.authenticate("Bearer " + "x" * 16385)


def test_rate_limiter_capacity_fails_closed(principal: Principal) -> None:
    limiter = RateLimiter(limit=1, capacity=1)
    limiter.check(principal)
    with pytest.raises(HTTPException) as error:
        limiter.check(principal.model_copy(update={"subject": "second"}))
    assert error.value.status_code == 503
