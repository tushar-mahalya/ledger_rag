import json
from typing import Any

import httpx
import pytest

from ledger_rag.security.workload_identity import (
    SecretRef,
    VaultWorkloadSettings,
    WorkloadSecretError,
    export_workload_secrets,
    fetch_workload_secrets,
)

JWT = "header.payload.signature"
SETTINGS = VaultWorkloadSettings(address="https://vault.example.test", role="ledger-evaluation")
REFS = {
    "LEDGER_QDRANT_API_KEY": SecretRef("ledger/qdrant"),
    "LEDGER_LLM_API_KEY": SecretRef("ledger/llm"),
}
STORE = {"ledger/qdrant": {"api_key": "qdrant-test-value"}, "ledger/llm": {"api_key": "llm-x"}}


class FakeVault:
    def __init__(self, *, login_status: int = 200, fail_reads: int = 0) -> None:
        self.calls: list[tuple[str, str, dict[str, str], Any]] = []
        self.login_status = login_status
        self.fail_reads = fail_reads

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, request.url.path, dict(request.headers), body))
        if request.url.path == "/v1/auth/jwt-modal/login":
            if self.login_status != 200:
                return httpx.Response(self.login_status, json={"errors": ["role mismatch"]})
            return httpx.Response(200, json={"auth": {"client_token": "hvs.short-lived"}})
        if request.url.path == "/v1/auth/token/revoke-self":
            return httpx.Response(204)
        if request.headers.get("x-vault-token") != "hvs.short-lived":
            return httpx.Response(403)
        if self.fail_reads:
            self.fail_reads -= 1
            return httpx.Response(503)
        path = request.url.path.removeprefix("/v1/secret/data/")
        if path not in STORE:
            return httpx.Response(404, json={"errors": []})
        return httpx.Response(200, json={"data": {"data": STORE[path]}})

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


async def test_fetch_logs_in_reads_each_path_once_and_revokes() -> None:
    vault = FakeVault()
    async with vault.client() as client:
        values = await fetch_workload_secrets(SETTINGS, REFS, identity_token=JWT, client=client)
    assert values == {"LEDGER_QDRANT_API_KEY": "qdrant-test-value", "LEDGER_LLM_API_KEY": "llm-x"}
    paths = [call[1] for call in vault.calls]
    assert paths == [
        "/v1/auth/jwt-modal/login",
        "/v1/secret/data/ledger/llm",
        "/v1/secret/data/ledger/qdrant",
        "/v1/auth/token/revoke-self",
    ]
    assert vault.calls[0][3] == {"role": "ledger-evaluation", "jwt": JWT}
    assert "x-vault-token" not in vault.calls[0][2]


async def test_namespace_header_is_sent_on_every_request() -> None:
    vault = FakeVault()
    settings = VaultWorkloadSettings(
        address="https://vault.example.test", role="r", namespace="admin/ledger"
    )
    async with vault.client() as client:
        await fetch_workload_secrets(
            settings, {"K": SecretRef("ledger/llm")}, identity_token=JWT, client=client
        )
    assert all(call[2]["x-vault-namespace"] == "admin/ledger" for call in vault.calls)


async def test_transient_read_failures_are_retried() -> None:
    vault = FakeVault(fail_reads=2)
    async with vault.client() as client:
        values = await fetch_workload_secrets(
            SETTINGS, {"K": SecretRef("ledger/llm")}, identity_token=JWT, client=client
        )
    assert values == {"K": "llm-x"}


async def test_rejected_login_reports_status_without_response_body() -> None:
    vault = FakeVault(login_status=403)
    async with vault.client() as client:
        with pytest.raises(WorkloadSecretError) as error:
            await fetch_workload_secrets(SETTINGS, REFS, identity_token=JWT, client=client)
    assert "HTTP 403" in str(error.value)
    assert "role mismatch" not in str(error.value)
    assert len(vault.calls) == 1


async def test_missing_field_fails_and_still_revokes_without_leaking_values() -> None:
    vault = FakeVault()
    refs = {**REFS, "OTHER": SecretRef("ledger/qdrant", "missing")}
    async with vault.client() as client:
        with pytest.raises(WorkloadSecretError) as error:
            await fetch_workload_secrets(SETTINGS, refs, identity_token=JWT, client=client)
    assert "ledger/qdrant#missing" in str(error.value)
    assert "qdrant-test-value" not in str(error.value)
    assert vault.calls[-1][1] == "/v1/auth/token/revoke-self"


async def test_unknown_path_fails_closed_and_revokes() -> None:
    vault = FakeVault()
    async with vault.client() as client:
        with pytest.raises(WorkloadSecretError, match="HTTP 404"):
            await fetch_workload_secrets(
                SETTINGS, {"K": SecretRef("ledger/absent")}, identity_token=JWT, client=client
            )
    assert vault.calls[-1][1] == "/v1/auth/token/revoke-self"


async def test_malformed_identity_token_is_rejected_before_network() -> None:
    vault = FakeVault()
    async with vault.client() as client:
        with pytest.raises(WorkloadSecretError, match="MODAL_IDENTITY_TOKEN"):
            await fetch_workload_secrets(SETTINGS, REFS, identity_token="", client=client)
    assert vault.calls == []


@pytest.mark.parametrize(
    "address",
    [
        "http://vault.example.test",
        "https://user:pass@vault.example.test",
        "https://vault.example.test/v1",
        "https://vault.example.test?x=1",
    ],
)
def test_settings_reject_unsafe_addresses(address: str) -> None:
    with pytest.raises(ValueError):
        VaultWorkloadSettings(address=address, role="r")


@pytest.mark.parametrize(
    ("path", "field"), [("../sys/raw", "api_key"), ("ledger/qdrant", "a-b"), ("", "api_key")]
)
def test_secret_refs_reject_traversal_and_bad_fields(path: str, field: str) -> None:
    with pytest.raises(ValueError):
        SecretRef(path, field)


async def test_export_is_noop_without_vault_address() -> None:
    environ = {"LEDGER_QDRANT_API_KEY": "from-modal-secret"}
    assert await export_workload_secrets(REFS, environ) is False
    assert environ == {"LEDGER_QDRANT_API_KEY": "from-modal-secret"}


def test_settings_require_role_when_address_set() -> None:
    with pytest.raises(ValueError, match="LEDGER_VAULT_ROLE"):
        VaultWorkloadSettings.from_env({"LEDGER_VAULT_ADDR": "https://vault.example.test"})


async def test_export_populates_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    vault = FakeVault()

    async def fake_fetch(
        settings: VaultWorkloadSettings, refs: Any, *, identity_token: str
    ) -> dict[str, str]:
        assert settings.role == "ledger-evaluation"
        async with vault.client() as client:
            return await fetch_workload_secrets(
                settings, refs, identity_token=identity_token, client=client
            )

    monkeypatch.setattr("ledger_rag.security.workload_identity.fetch_workload_secrets", fake_fetch)
    environ = {
        "LEDGER_VAULT_ADDR": "https://vault.example.test/",
        "LEDGER_VAULT_ROLE": "ledger-evaluation",
        "MODAL_IDENTITY_TOKEN": JWT,
    }
    assert await export_workload_secrets(REFS, environ) is True
    assert environ["LEDGER_LLM_API_KEY"] == "llm-x"
