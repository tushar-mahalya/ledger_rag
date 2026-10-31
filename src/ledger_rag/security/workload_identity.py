"""Exchange a Modal workload identity token for short-lived Vault secrets.

Modal injects ``MODAL_IDENTITY_TOKEN`` (an RS256 JWT issued by https://oidc.modal.com)
into every container. Vault's JWT auth method verifies it against Modal's JWKS and issues a
minutes-long token bound to a per-workload policy, so provider credentials never have to
be copied into Modal Secrets. When ``LEDGER_VAULT_ADDR`` is unset, callers keep using
environment variables projected by ``scripts/sync_modal_secrets.sh``.
"""

from __future__ import annotations

import asyncio
import os
import random
import re
import ssl
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx
import structlog

logger = structlog.get_logger()

IDENTITY_TOKEN_ENV = "MODAL_IDENTITY_TOKEN"
_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_PATH = re.compile(r"^[A-Za-z0-9_-]{1,64}(/[A-Za-z0-9_-]{1,64}){0,7}$")
_FIELD = re.compile(r"^[A-Za-z0-9_]{1,64}$")
_ATTEMPTS = 3


class WorkloadSecretError(RuntimeError):
    """Secret retrieval failed; messages never include secret material or response bodies."""


@dataclass(frozen=True)
class SecretRef:
    path: str
    field: str = "api_key"

    def __post_init__(self) -> None:
        if not _PATH.fullmatch(self.path) or not _FIELD.fullmatch(self.field):
            raise ValueError("Invalid Vault KV path or field")


@dataclass(frozen=True)
class VaultWorkloadSettings:
    address: str
    role: str
    auth_mount: str = "jwt-modal"
    kv_mount: str = "secret"
    ca_pem: str | None = None
    namespace: str | None = None
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        parsed = urlsplit(self.address)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise ValueError("Vault address must be a credential-free HTTPS origin")
        if not all(_NAME.fullmatch(value) for value in (self.role, self.auth_mount, self.kv_mount)):
            raise ValueError("Invalid Vault role or mount name")
        if self.namespace is not None and not _PATH.fullmatch(self.namespace):
            raise ValueError("Invalid Vault namespace")
        if not 0 < self.timeout_seconds <= 60:
            raise ValueError("Vault timeout must be in (0, 60] seconds")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> VaultWorkloadSettings | None:
        address = environ.get("LEDGER_VAULT_ADDR", "").strip()
        if not address:
            return None
        role = environ.get("LEDGER_VAULT_ROLE", "").strip()
        if not role:
            raise ValueError("LEDGER_VAULT_ROLE is required when LEDGER_VAULT_ADDR is set")
        return cls(
            address=address.rstrip("/"),
            role=role,
            auth_mount=environ.get("LEDGER_VAULT_AUTH_MOUNT", "jwt-modal"),
            kv_mount=environ.get("LEDGER_VAULT_KV_MOUNT", "secret"),
            ca_pem=environ.get("LEDGER_VAULT_CA_PEM") or None,
            namespace=environ.get("LEDGER_VAULT_NAMESPACE") or None,
        )

    def verify(self) -> ssl.SSLContext | bool:
        if self.ca_pem is None:
            return True
        return ssl.create_default_context(cadata=self.ca_pem)


def _retryable(error: Exception) -> bool:
    if isinstance(error, httpx.TransportError):
        return True
    return isinstance(error, httpx.HTTPStatusError) and error.response.status_code in (
        429,
        500,
        502,
        503,
        504,
    )


async def _request(
    client: httpx.AsyncClient, method: str, url: str, **kwargs: Any
) -> dict[str, Any]:
    for attempt in range(_ATTEMPTS):
        try:
            response = await client.request(method, url, **kwargs)
            response.raise_for_status()
            body = response.json() if response.content else {}
            if not isinstance(body, dict):
                raise WorkloadSecretError("Vault returned a non-object response")
            return body
        except (httpx.TransportError, httpx.HTTPStatusError) as error:
            if attempt + 1 == _ATTEMPTS or not _retryable(error):
                status = (
                    error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
                )
                raise WorkloadSecretError(
                    f"Vault {method} {urlsplit(url).path} failed"
                    + (f" with HTTP {status}" if status else f": {type(error).__name__}")
                ) from None
            await asyncio.sleep(random.uniform(0, 0.25 * 2**attempt))
        except ValueError:
            raise WorkloadSecretError("Vault returned invalid JSON") from None
    raise AssertionError("unreachable")


async def fetch_workload_secrets(
    settings: VaultWorkloadSettings,
    refs: Mapping[str, SecretRef],
    *,
    identity_token: str,
    client: httpx.AsyncClient | None = None,
) -> dict[str, str]:
    """Log in with the workload JWT, read each referenced KV v2 field, then revoke the token."""
    if identity_token.count(".") != 2:
        raise WorkloadSecretError(f"{IDENTITY_TOKEN_ENV} is missing or malformed")
    namespace = {"X-Vault-Namespace": settings.namespace} if settings.namespace else {}
    owned = client is None
    http = client or httpx.AsyncClient(verify=settings.verify(), timeout=settings.timeout_seconds)
    base = settings.address.rstrip("/")
    try:
        login = await _request(
            http,
            "POST",
            f"{base}/v1/auth/{settings.auth_mount}/login",
            headers=namespace,
            json={"role": settings.role, "jwt": identity_token},
        )
        auth = login.get("auth")
        token = auth.get("client_token") if isinstance(auth, dict) else None
        if not isinstance(token, str) or not token:
            raise WorkloadSecretError("Vault login returned no client token")
        vault_headers = {**namespace, "X-Vault-Token": token}
        try:
            records: dict[str, dict[str, Any]] = {}
            for path in sorted({ref.path for ref in refs.values()}):
                body = await _request(
                    http,
                    "GET",
                    f"{base}/v1/{settings.kv_mount}/data/{path}",
                    headers=vault_headers,
                )
                data = body.get("data")
                record = data.get("data") if isinstance(data, dict) else None
                if not isinstance(record, dict):
                    raise WorkloadSecretError(f"Vault path {path} has no KV v2 data")
                records[path] = record
            values: dict[str, str] = {}
            for name, ref in refs.items():
                value = records[ref.path].get(ref.field)
                if not isinstance(value, str) or not value.strip():
                    raise WorkloadSecretError(f"Vault field {ref.path}#{ref.field} is empty")
                values[name] = value.strip()
            return values
        finally:
            try:
                await _request(
                    http, "POST", f"{base}/v1/auth/token/revoke-self", headers=vault_headers
                )
            except WorkloadSecretError:
                logger.warning("vault_token_revoke_failed")
    finally:
        if owned:
            await http.aclose()


async def export_workload_secrets(
    refs: Mapping[str, SecretRef], environ: MutableMapping[str, str] | None = None
) -> bool:
    """Populate ``environ`` from Vault when workload identity is configured.

    Returns ``False`` (leaving ``environ`` untouched) when ``LEDGER_VAULT_ADDR`` is unset so
    callers fall back to Modal-Secret environment variables.
    """
    target = os.environ if environ is None else environ
    settings = VaultWorkloadSettings.from_env(target)
    if settings is None:
        return False
    values = await fetch_workload_secrets(
        settings, refs, identity_token=target.get(IDENTITY_TOKEN_ENV, "")
    )
    target.update(values)
    logger.info("vault_workload_secrets_loaded", role=settings.role, count=len(values))
    return True
