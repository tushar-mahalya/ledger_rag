"""Deliver manifest-routed secrets to runtime stores (Modal Secrets, Cloudflare Workers).

Each remote target receives exactly the variables the manifest routes to it plus the
non-secret ``LEDGER_SECRETS_REVISION``. Delivery replaces the target's contents so a
key removed from the manifest is also removed remotely. Operator credentials
(Modal/Cloudflare tokens) authenticate the delivery and are never delivered.
"""

import importlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from ledgerctl.manifest import REVISION_KEY, Environment, Manifest, Target
from ledgerctl.process import OpsError, Runner, check, run

# Pinned; `secret bulk` deletes keys given JSON null values from 4.97 onwards.
WRANGLER = "wrangler@4.151.3"


@dataclass(frozen=True)
class Delivery:
    target: Target
    payload: Mapping[str, str]

    @property
    def variables(self) -> list[str]:
        return sorted(self.payload)


def plan(
    manifest: Manifest,
    env: str,
    values: Mapping[str, str],
    revision: str,
    targets: list[str] | None = None,
    *,
    strict: bool = True,
) -> list[Delivery]:
    """Payloads for the remote targets in ``env`` (optionally a subset)."""
    names = targets or sorted(name for name, t in manifest.targets.items() if t.remote)
    deliveries = []
    for name in names:
        target = manifest.target(name)
        if not target.remote:
            raise OpsError(f"{name} is a {target.platform} target and is never synced")
        payload = manifest.payload(env, name, values, strict=strict)
        if payload:
            deliveries.append(Delivery(target, {**payload, REVISION_KEY: revision}))
    return deliveries


class Publisher(Protocol):
    def publish(self, environment: Environment, delivery: Delivery) -> None: ...


class ModalPublisher:
    """Replaces a named Modal Secret in the mapped Modal environment.

    Authentication uses ``MODAL_TOKEN_ID``/``MODAL_TOKEN_SECRET`` from ``credentials``,
    exported into this process only, before the SDK is imported.
    """

    def __init__(self, credentials: Mapping[str, str]) -> None:
        for name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
            if not credentials.get(name):
                raise OpsError(f"{name} is not set for this environment; add it with ledgerctl set")
            os.environ[name] = credentials[name]
        self._modal: Any = None

    def _sdk(self) -> Any:
        if self._modal is None:
            # Resolved at runtime: ledgerctl stays stdlib-only, Modal is the optional
            # workspace `compute` group, and the SDK is used untyped (Any) here.
            try:
                modal = importlib.import_module("modal")
            except ImportError:
                raise OpsError(
                    "The Modal SDK is missing: run with `uv run --group compute`"
                ) from None
            self._modal = modal
        return self._modal

    def publish(self, environment: Environment, delivery: Delivery) -> None:
        modal = self._sdk()
        name = delivery.target.resource
        env = environment.modal_environment
        try:
            modal.Secret.objects.delete(name, allow_missing=True, environment_name=env)
            modal.Secret.objects.create(name, dict(delivery.payload), environment_name=env)
        except Exception as exc:  # SDK errors may carry request context; keep the class only.
            raise OpsError(
                f"Modal secret {name} in environment {env} failed: {type(exc).__name__}. "
                f"Does the environment exist? (ledgerctl modal environment create {env})"
            ) from None


class CloudflarePublisher:
    """Replaces a Worker's secrets with ``wrangler secret bulk`` (JSON on stdin)."""

    def __init__(self, credentials: Mapping[str, str], runner: Runner = run) -> None:
        for name in ("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"):
            if not credentials.get(name):
                raise OpsError(f"{name} is not set for this environment; add it with ledgerctl set")
        self._env = {
            **os.environ,
            "CLOUDFLARE_API_TOKEN": credentials["CLOUDFLARE_API_TOKEN"],
            "CLOUDFLARE_ACCOUNT_ID": credentials["CLOUDFLARE_ACCOUNT_ID"],
        }
        self._runner = runner

    def _wrangler(self, *args: str) -> list[str]:
        return ["npx", "--yes", WRANGLER, *args]

    def publish(self, environment: Environment, delivery: Delivery) -> None:
        worker = f"{delivery.target.resource}-{environment.cloudflare_env}"
        listed = self._runner(
            self._wrangler("secret", "list", "--name", worker, "--format", "json"),
            None,
            self._env,
            None,
        )
        check(listed, f"Listing secrets of Worker {worker}")
        try:
            existing = {item["name"] for item in json.loads(listed.stdout)}
        except (ValueError, TypeError, KeyError):
            raise OpsError(f"Unexpected secret listing for Worker {worker}") from None
        body: dict[str, str | None] = {name: None for name in existing - set(delivery.payload)}
        body.update(delivery.payload)
        result = self._runner(
            self._wrangler("secret", "bulk", "--name", worker),
            json.dumps(body).encode(),
            self._env,
            None,
        )
        check(result, f"Updating secrets of Worker {worker}", diagnostic=False)
