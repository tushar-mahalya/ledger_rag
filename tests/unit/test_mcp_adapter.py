import asyncio
import time
from datetime import date
from types import SimpleNamespace

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from fastapi.testclient import TestClient

from ledger_rag.api.app import create_app
from ledger_rag.core.config import Settings
from ledger_rag.core.models import Document, Principal
from ledger_rag.ingestion.chunking import chunk_document
from ledger_rag.mcp_adapter.server import MCPAdapter
from ledger_rag.retrieval.memory import MemoryRetriever
from tests.unit.external_fakes import external_data
from tests.unit.test_core import FakeProvider

HEADERS = {
    "Authorization": "Bearer test-only-token",
    "Accept": "application/json, text/event-stream",
    "MCP-Protocol-Version": "2025-11-25",
}


def enabled(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "mcp_enabled": True,
            "mcp_resource_url": "http://testserver/mcp",
            "mcp_allowed_origins": ["http://testserver"],
        }
    )


def rpc(client: TestClient, method: str, params: dict | None = None) -> dict:
    response = client.post(
        "/mcp",
        headers=HEADERS,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_mcp_disabled_and_resource_metadata(settings: Settings) -> None:
    with TestClient(create_app(settings, MemoryRetriever(), FakeProvider([]))) as client:
        assert client.post("/mcp").status_code == 404
        assert client.get("/.well-known/oauth-protected-resource").status_code == 404
    with TestClient(create_app(enabled(settings), MemoryRetriever(), FakeProvider([]))) as client:
        response = client.post("/mcp", json={})
        assert response.status_code == 401
        assert "resource_metadata" in response.headers["www-authenticate"]
        metadata = client.get("/.well-known/oauth-protected-resource/mcp").json()
        assert metadata["resource"] == "http://testserver/mcp"
        assert metadata["bearer_methods_supported"] == ["header"]
        assert (
            client.post(
                "/mcp", headers={**HEADERS, "Origin": "https://evil.example"}, json={}
            ).status_code
            == 403
        )
        assert client.post(
            "/mcp", headers={**HEADERS, "Host": "evil.example"}, json={}
        ).status_code in {400, 421}


def test_three_readonly_tools_and_protocol(settings: Settings) -> None:
    with TestClient(create_app(enabled(settings), MemoryRetriever(), FakeProvider([]))) as client:
        initialized = rpc(
            client,
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "offline-test", "version": "1"},
            },
        )
        assert initialized["result"]["serverInfo"]["name"] == "Ledger evidence"
        listed = rpc(client, "tools/list")["result"]["tools"]
        assert {tool["name"] for tool in listed} == {
            "search_evidence",
            "lookup_company_facts",
            "lookup_macro_series",
        }
        assert all(tool["annotations"]["readOnlyHint"] for tool in listed)
        assert all(not tool["annotations"]["destructiveHint"] for tool in listed)
        assert rpc(client, "tools/call", {"name": "delete_documents"})["result"]["isError"]
        result = rpc(
            client,
            "tools/call",
            {
                "name": "lookup_company_facts",
                "arguments": {"company": "AAPL", "concepts": ["Assets"]},
            },
        )
        assert result["result"]["isError"]


def test_mcp_search_filters_and_external_cutoff(settings: Settings) -> None:
    store = MemoryRetriever()
    docs = [
        Document(
            id=name,
            tenant_id=tenant,
            title=name,
            text="Revenue was USD 12 million.",
            source="urn:test",
            published_at=published,
            allowed_groups=groups,
        )
        for name, tenant, published, groups in [
            ("public", "demo", date(2024, 1, 1), []),
            ("other-tenant", "other", date(2024, 1, 1), []),
            ("private", "demo", date(2024, 1, 1), ["secret"]),
            ("future", "demo", date(2026, 1, 1), []),
        ]
    ]
    asyncio.run(store.upsert([c for doc in docs for c in chunk_document(doc)]))
    with TestClient(create_app(enabled(settings), store, FakeProvider([]))) as client:
        result = rpc(
            client,
            "tools/call",
            {
                "name": "search_evidence",
                "arguments": {"question": "Revenue", "as_of": "2025-01-01"},
            },
        )
        assert "public" in result["result"]["content"][0]["text"]
        for forbidden in ["other-tenant", "private", "future"]:
            assert forbidden not in result["result"]["content"][0]["text"]
        data = external_data()
        client.app.state.runtime.service.external = data
        external = rpc(
            client,
            "tools/call",
            {
                "name": "lookup_macro_series",
                "arguments": {
                    "series_id": "FEDFUNDS",
                    "dates": ["2023-08-01"],
                    "as_of": "2023-09-01",
                },
            },
        )
        assert not external["result"].get("isError", False), external
        assert '"as_of": "2023-09-01"' in external["result"]["content"][0]["text"]
        assert '"provenance"' in external["result"]["content"][0]["text"]
        rejected = rpc(
            client,
            "tools/call",
            {"name": "search_evidence", "arguments": {"question": "Revenue", "tenant_id": "other"}},
        )
        assert rejected["result"]["isError"]


async def test_mcp_principal_context_is_request_isolated(settings: Settings) -> None:
    app = create_app(enabled(settings), MemoryRetriever(), FakeProvider([]))
    async with app.router.lifespan_context(app):

        class Auth:
            async def authenticate(self, authorization: str | None) -> Principal:
                assert authorization is not None
                return Principal(subject=authorization, tenant_id=authorization, roles=["reader"])

        app.state.mcp.auth = Auth()

        async def retrieve(query: object, principal: Principal) -> tuple[list, float]:
            await asyncio.sleep(0.001)
            assert principal.tenant_id in {"alpha", "beta"}
            from ledger_rag.core.models import Hit

            doc = Document(
                id=principal.tenant_id,
                tenant_id=principal.tenant_id,
                title=principal.tenant_id,
                text="Revenue 12",
                source="urn:test",
                published_at=date(2024, 1, 1),
            )
            return [Hit(chunk=chunk_document(doc)[0], score=1)], 0

        app.state.runtime.service.retrieve = retrieve
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            responses = await asyncio.gather(
                *(
                    client.post(
                        "/mcp",
                        headers={**HEADERS, "Authorization": tenant},
                        json={
                            "jsonrpc": "2.0",
                            "id": i,
                            "method": "tools/call",
                            "params": {
                                "name": "search_evidence",
                                "arguments": {"question": "Revenue"},
                            },
                        },
                    )
                    for i, tenant in enumerate(["alpha", "beta"])
                )
            )
        for response, tenant in zip(responses, ["alpha", "beta"], strict=True):
            assert response.status_code == 200, response.text
            assert f'"title": "{tenant}"' in response.json()["result"]["content"][0]["text"]
        assert app.state.runtime.slots._value == settings.max_concurrent_queries


async def test_oidc_mcp_audience_scopes_and_expiration(settings: Settings) -> None:
    configured = enabled(settings).model_copy(
        update={
            "auth_mode": "oidc",
            "oidc_issuer": "https://issuer.example",
            "oidc_audience": "ledger-api",
            "oidc_jwks_url": "https://issuer.example/jwks",
            "mcp_required_scopes": ["ledger:read"],
        }
    )
    adapter = MCPAdapter(configured)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    adapter.auth.jwks = SimpleNamespace(
        get_signing_key_from_jwt=lambda token: SimpleNamespace(key=key.public_key())
    )

    def token(**changes: object) -> str:
        return jwt.encode(
            {
                "iss": configured.oidc_issuer,
                "aud": configured.mcp_resource_url,
                "sub": "analyst",
                "tenant_id": "alpha",
                "roles": ["reader"],
                "groups": ["finance"],
                "scope": "ledger:read",
                "iat": int(time.time()),
                "exp": int(time.time()) + 60,
                **changes,
            },
            key,
            algorithm="RS256",
        )

    principal = await adapter.auth.authenticate("Bearer " + token())
    assert principal.tenant_id == "alpha" and principal.groups == ["finance"]
    for changes, code in [
        ({"aud": "ledger-api"}, 401),
        ({"scope": ""}, 403),
        ({"scope": ["ledger:read"]}, 403),
        ({"exp": 0}, 401),
    ]:
        with pytest.raises(HTTPException) as error:
            await adapter.auth.authenticate("Bearer " + token(**changes))
        assert error.value.status_code == code
    assert adapter.metadata()["authorization_servers"] == ["https://issuer.example"]


async def test_mcp_deadline_cancels_work_releases_slots_and_sanitizes_failures(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    configured = enabled(settings).model_copy(update={"query_deadline_seconds": 0.2})
    app = create_app(configured, MemoryRetriever(), FakeProvider([]))
    cancelled = asyncio.Event()
    async with app.router.lifespan_context(app):

        async def slow(query: object, principal: Principal) -> tuple[list, float]:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return [], 0

        app.state.runtime.service.retrieve = slow
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "search_evidence", "arguments": {"question": "Revenue"}},
        }
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post("/mcp", headers=HEADERS, json=body)
            assert response.status_code == 504 or response.json()["result"]["isError"]
            assert cancelled.is_set()
            assert app.state.runtime.slots._value == settings.max_concurrent_queries

            async def broken(query: object, principal: Principal) -> tuple[list, float]:
                raise RuntimeError("private-document-and-secret-value")

            app.state.runtime.service.retrieve = broken
            response = await client.post("/mcp", headers=HEADERS, json=body)
            assert response.json()["result"]["isError"]
            assert "private-document-and-secret-value" not in response.text
            logs = capsys.readouterr()
            assert "private-document-and-secret-value" not in logs.out + logs.err
            assert app.state.runtime.slots._value == settings.max_concurrent_queries


def test_mcp_rejects_oversize_body(settings: Settings) -> None:
    with TestClient(create_app(enabled(settings), MemoryRetriever(), FakeProvider([]))) as client:
        response = client.post("/mcp", headers=HEADERS, content="x" * 65537)
        assert response.status_code == 413
