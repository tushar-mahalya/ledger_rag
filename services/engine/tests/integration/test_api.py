from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from ledger_rag.api.app import create_app
from ledger_rag.core.config import Settings
from ledger_rag.core.models import Document
from ledger_rag.generation.providers import make_provider
from ledger_rag.retrieval.memory import MemoryRetriever

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


def test_blank_replacement_is_rejected_before_batch_mutation(
    client: TestClient, document: Document
) -> None:
    original = document.model_dump(mode="json")
    assert (
        client.post("/v1/documents", json={"documents": [original]}, headers=AUTH).status_code
        == 201
    )
    invalid = {**original, "text": " \n\t ", "allowed_groups": ["private"]}
    assert (
        client.post("/v1/documents", json={"documents": [invalid]}, headers=AUTH).status_code == 422
    )
    response = client.post("/v1/query", json={"question": "Revenue"}, headers=AUTH)
    assert not response.json()["abstained"]


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


def test_removed_secret_file_directory_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    from ledger_rag.core.config import get_settings

    monkeypatch.setenv("LEDGER_SECRETS_DIR", "/run/ledger-secrets")
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="no longer supported"):
            get_settings()
    finally:
        get_settings.cache_clear()


class BootstrappingRetriever(MemoryRetriever):
    def __init__(self) -> None:
        super().__init__()
        self.bootstraps = 0

    async def bootstrap(self) -> None:
        self.bootstraps += 1


@pytest.mark.parametrize("enabled", [False, True])
def test_collection_bootstrap_is_an_explicit_startup_step(
    settings: Settings, enabled: bool
) -> None:
    store = BootstrappingRetriever()
    configured = settings.model_copy(update={"qdrant_bootstrap_on_startup": enabled})
    with TestClient(create_app(configured, store, make_provider(configured))) as test_client:
        assert store.bootstraps == int(enabled)
        for _ in range(3):
            assert test_client.get("/health/ready").status_code == 200
    assert store.bootstraps == int(enabled)


def test_retriever_collectors_are_exported_on_metrics(settings: Settings) -> None:
    from prometheus_client import Counter

    rejected = Counter("ledger_qdrant_rejected_points", "test", ["reason"], registry=None)
    rejected.labels("invalid_payload").inc(2)

    class InstrumentedRetriever(MemoryRetriever):
        def collectors(self) -> list[Counter]:
            return [rejected]

    app = create_app(settings, InstrumentedRetriever(), make_provider(settings))
    with TestClient(app) as test_client:
        body = test_client.get("/metrics", headers=AUTH).text
    assert 'ledger_qdrant_rejected_points_total{reason="invalid_payload"} 2.0' in body


def test_document_acl_changes_apply_immediately_and_delete_erases(
    client: TestClient, document: Document
) -> None:
    body = {"documents": [document.model_dump(mode="json")]}
    assert client.post("/v1/documents", json=body, headers=AUTH).status_code == 201

    def cited() -> bool:
        answer = client.post("/v1/query", json={"question": "Revenue"}, headers=AUTH).json()
        return any(c["document_id"] == document.id for c in answer["citations"])

    assert cited()
    url = f"/v1/documents/{document.id}/acl"
    # The demo principal holds only the "analyst" group: restricting to "legal" hides it.
    restricted = client.patch(url, json={"allowed_groups": ["legal"]}, headers=AUTH)
    assert restricted.status_code == 200 and restricted.json()["chunks"] >= 1
    assert not cited()
    assert client.patch(url, json={"allowed_groups": []}, headers=AUTH).status_code == 200
    assert cited()
    assert (
        client.patch(
            url, json={"allowed_groups": ["x"], "tenant_id": "other"}, headers=AUTH
        ).status_code
        == 422
    )
    assert (
        client.patch(
            "/v1/documents/missing/acl", json={"allowed_groups": []}, headers=AUTH
        ).status_code
        == 404
    )
    assert client.patch(url, json={"allowed_groups": []}).status_code == 401
    assert client.delete(f"/v1/documents/{document.id}", headers=AUTH).status_code == 200
    assert not cited()
    assert client.delete(f"/v1/documents/{document.id}", headers=AUTH).status_code == 404
