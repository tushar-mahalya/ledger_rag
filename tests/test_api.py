from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ledger_rag.api import create_app
from ledger_rag.config import Settings
from ledger_rag.models import Document
from ledger_rag.providers import make_provider
from ledger_rag.retrieval import MemoryRetriever

AUTH = {"Authorization": "Bearer test-only-token"}


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings, MemoryRetriever(), make_provider(settings))
    with TestClient(app) as test_client:
        yield test_client


def test_health_auth_ingest_and_query(client: TestClient, document: Document) -> None:
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 200
    assert client.post("/v1/query", json={"question": "Revenue"}).status_code == 401
    assert (
        client.post(
            "/v1/documents", json={"documents": [document.model_dump(mode="json")]}, headers=AUTH
        ).status_code
        == 201
    )
    response = client.post("/v1/query", json={"question": "Revenue"}, headers=AUTH)
    assert response.status_code == 200
    assert not response.json()["abstained"]
    assert response.json()["citations"][0]["document_id"] == document.id
    assert response.headers["x-request-id"] == response.json()["request_id"]
    assert client.get("/metrics").status_code == 401
    assert "ledger_http_requests_total" in client.get("/metrics", headers=AUTH).text


def test_cross_tenant_and_duplicate_ingest_rejected(client: TestClient, document: Document) -> None:
    other = document.model_copy(update={"tenant_id": "other"})
    assert (
        client.post(
            "/v1/documents", json={"documents": [other.model_dump(mode="json")]}, headers=AUTH
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/v1/documents",
            json={"documents": [document.model_dump(mode="json")] * 2},
            headers=AUTH,
        ).status_code
        == 422
    )


def test_stream_has_validated_answer_and_done(client: TestClient, document: Document) -> None:
    client.post(
        "/v1/documents", json={"documents": [document.model_dump(mode="json")]}, headers=AUTH
    )
    with client.stream(
        "POST", "/v1/query/stream", json={"question": "Revenue", "profile": "voice"}, headers=AUTH
    ) as response:
        body = response.read().decode()
        assert response.status_code == 200
        assert "event: evidence" in body
        assert "event: answer" in body
        assert "event: done" in body
        assert "event: error" not in body


def test_empty_evidence_abstains(client: TestClient) -> None:
    response = client.post("/v1/query", json={"question": "Unknown topic"}, headers=AUTH)
    assert response.json()["abstained"]
    assert not response.json()["citations"]


def test_query_validation_and_body_limit(client: TestClient) -> None:
    assert client.post("/v1/query", json={"question": "  "}, headers=AUTH).status_code == 422
    assert (
        client.post(
            "/v1/query", json={"question": "Revenue", "top_k": 11}, headers=AUTH
        ).status_code
        == 422
    )
    response = client.post(
        "/v1/query",
        content=b"x" * 2_000_001,
        headers={**AUTH, "Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_rate_limit(settings: Settings) -> None:
    limited = settings.model_copy(update={"rate_limit_per_minute": 1})
    with TestClient(create_app(limited)) as client:
        assert (
            client.post("/v1/query", json={"question": "Revenue"}, headers=AUTH).status_code == 200
        )
        assert (
            client.post("/v1/query", json={"question": "Revenue"}, headers=AUTH).status_code == 429
        )


def test_production_fails_closed() -> None:
    with pytest.raises(ValueError):
        Settings(environment="production")


@pytest.mark.parametrize(
    "endpoint",
    ["file:///etc/passwd", "https://user:secret@example.com", "https://api.test?key=secret"],
)
def test_endpoint_credentials_are_not_accepted(endpoint: str) -> None:
    with pytest.raises(ValueError) as error:
        Settings(demo_token="test-only-token", llm_base_url=endpoint)
    assert "test-only-token" not in str(error.value)


def test_vault_secret_file(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ledger_rag.config import get_settings

    (tmp_path / "LEDGER_DEMO_TOKEN").write_text("file-secret")
    monkeypatch.setenv("LEDGER_SECRETS_DIR", str(tmp_path))
    monkeypatch.delenv("LEDGER_DEMO_TOKEN", raising=False)
    get_settings.cache_clear()
    try:
        loaded = get_settings()
        assert loaded.demo_token and loaded.demo_token.get_secret_value() == "file-secret"
    finally:
        get_settings.cache_clear()
