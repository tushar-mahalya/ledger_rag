"""The public secret manifest: names, environments and delivery routes, never values.

``config/secrets.toml`` declares every credential Ledger uses, which environments hold
it and which runtime receives it under which variable name. Values live only in the
private SOPS + age repository; this module validates routing and builds per-target
payloads from decrypted values in memory.
"""

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Platform = Literal["operator", "local", "modal", "cloudflare"]
Generator = Literal["hex32"]

NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
RESOURCE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
ENVIRONMENT = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
PLATFORMS: frozenset[str] = frozenset({"operator", "local", "modal", "cloudflare"})
GENERATORS: frozenset[str] = frozenset({"hex32"})
# Delivered with every remote payload so a runtime can report which revision it holds.
REVISION_KEY = "LEDGER_SECRETS_REVISION"


class ManifestError(ValueError):
    """The manifest or a requested environment/target is invalid."""


@dataclass(frozen=True)
class Environment:
    name: str
    modal_environment: str
    cloudflare_env: str


@dataclass(frozen=True)
class Target:
    name: str
    platform: Platform
    # Modal Secret name or Cloudflare Worker name; empty for operator/local.
    resource: str = ""

    @property
    def remote(self) -> bool:
        return self.platform in {"modal", "cloudflare"}


@dataclass(frozen=True)
class SecretSpec:
    name: str
    description: str
    envs: frozenset[str]
    # target name -> variable names delivered to that target
    deliver: Mapping[str, tuple[str, ...]]
    required: bool = True
    generate: Generator | None = None
    # A public constant (e.g. a documented base URL); never stored in SOPS.
    value: str | None = None

    @property
    def stored(self) -> bool:
        return self.value is None


@dataclass(frozen=True)
class Manifest:
    environments: Mapping[str, Environment]
    targets: Mapping[str, Target]
    secrets: Mapping[str, SecretSpec] = field(default_factory=dict)

    def environment(self, name: str) -> Environment:
        try:
            return self.environments[name]
        except KeyError:
            raise ManifestError(f"Unknown environment {name!r}") from None

    def target(self, name: str) -> Target:
        try:
            return self.targets[name]
        except KeyError:
            raise ManifestError(f"Unknown target {name!r}") from None

    def secret(self, name: str) -> SecretSpec:
        try:
            return self.secrets[name]
        except KeyError:
            raise ManifestError(f"{name} is not declared in the secret manifest") from None

    def stored_names(self, env: str) -> list[str]:
        self.environment(env)
        return sorted(s.name for s in self.secrets.values() if s.stored and env in s.envs)

    def missing(self, env: str, values: Mapping[str, str], target: str | None = None) -> list[str]:
        """Required stored secrets for ``env`` (optionally routed to ``target``) without a value."""
        return [
            name
            for name in self.stored_names(env)
            if self.secrets[name].required
            and not values.get(name)
            and (target is None or target in self.secrets[name].deliver)
        ]

    def unknown(self, env: str, values: Mapping[str, str]) -> list[str]:
        """Stored keys that the manifest does not declare for ``env``."""
        declared = set(self.stored_names(env))
        return sorted(name for name in values if name not in declared)

    def payload(
        self, env: str, target: str, values: Mapping[str, str], *, strict: bool = True
    ) -> dict[str, str]:
        """Variables delivered to ``target`` in ``env``.

        Absent optional values are skipped; an absent required value raises unless
        ``strict`` is false (local runs and dry runs report it instead).
        """
        self.environment(env)
        self.target(target)
        payload: dict[str, str] = {}
        for spec in self.secrets.values():
            if env not in spec.envs or target not in spec.deliver:
                continue
            value = spec.value if spec.value is not None else values.get(spec.name, "")
            if not value:
                if spec.required and strict:
                    raise ManifestError(f"{spec.name} has no value in {env}")
                continue
            for variable in spec.deliver[target]:
                payload[variable] = value
        return payload


def _strings(value: Any, where: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ManifestError(f"{where} must be a list of strings")
    return value


def _environment(name: str, raw: Any) -> Environment:
    if not ENVIRONMENT.fullmatch(name) or not isinstance(raw, dict):
        raise ManifestError(f"Invalid environment {name!r}")
    modal_env = raw.get("modal_environment", name)
    cloudflare_env = raw.get("cloudflare_env", name)
    for label, value in (("modal_environment", modal_env), ("cloudflare_env", cloudflare_env)):
        if not isinstance(value, str) or not ENVIRONMENT.fullmatch(value):
            raise ManifestError(f"Environment {name}: invalid {label}")
    return Environment(name, modal_env, cloudflare_env)


def _target(name: str, raw: Any) -> Target:
    if not RESOURCE.fullmatch(name) or not isinstance(raw, dict):
        raise ManifestError(f"Invalid target {name!r}")
    platform = raw.get("platform")
    if platform not in PLATFORMS:
        raise ManifestError(f"Target {name}: platform must be one of {sorted(PLATFORMS)}")
    resource = raw.get("secret") if platform == "modal" else raw.get("worker")
    if platform in {"modal", "cloudflare"}:
        if not isinstance(resource, str) or not RESOURCE.fullmatch(resource):
            field_name = "secret" if platform == "modal" else "worker"
            raise ManifestError(f"Target {name}: {field_name} must be a lowercase resource name")
        return Target(name, platform, resource)
    return Target(name, platform)


def _secret(
    name: str, raw: Any, environments: Mapping[str, Environment], targets: Mapping[str, Target]
) -> SecretSpec:
    if not NAME.fullmatch(name) or not isinstance(raw, dict):
        raise ManifestError(f"Invalid secret name {name!r}")
    unknown_fields = set(raw) - {"description", "envs", "deliver", "required", "generate", "value"}
    if unknown_fields:
        raise ManifestError(f"{name}: unknown fields {sorted(unknown_fields)}")
    description = raw.get("description", "")
    if not isinstance(description, str) or not description:
        raise ManifestError(f"{name}: description is required")
    envs = frozenset(_strings(raw.get("envs", sorted(environments)), f"{name}.envs"))
    if not envs or not envs <= set(environments):
        raise ManifestError(f"{name}: envs must name declared environments")
    raw_deliver = raw.get("deliver")
    if not isinstance(raw_deliver, dict) or not raw_deliver:
        raise ManifestError(f"{name}: deliver must map at least one target")
    deliver: dict[str, tuple[str, ...]] = {}
    for target, alias in raw_deliver.items():
        if target not in targets:
            raise ManifestError(f"{name}: unknown target {target!r}")
        if alias is True:
            variables = [name]
        elif isinstance(alias, str):
            variables = [alias]
        else:
            variables = _strings(alias, f"{name}.deliver.{target}")
        if not variables or not all(NAME.fullmatch(v) for v in variables):
            raise ManifestError(f"{name}: invalid variable name for target {target}")
        if REVISION_KEY in variables:
            raise ManifestError(f"{name}: {REVISION_KEY} is reserved")
        deliver[target] = tuple(variables)
    required = raw.get("required", True)
    generate = raw.get("generate")
    value = raw.get("value")
    if not isinstance(required, bool):
        raise ManifestError(f"{name}: required must be a boolean")
    if generate is not None and generate not in GENERATORS:
        raise ManifestError(f"{name}: generate must be one of {sorted(GENERATORS)}")
    if value is not None and (not isinstance(value, str) or not value or generate is not None):
        raise ManifestError(f"{name}: value must be a non-empty public constant")
    operator_only = all(targets[t].platform == "operator" for t in deliver)
    if any(targets[t].platform == "operator" for t in deliver) and not operator_only:
        raise ManifestError(f"{name}: operator credentials cannot be delivered to runtimes")
    return SecretSpec(name, description, envs, deliver, required, generate, value)


def parse_manifest(raw: Mapping[str, Any]) -> Manifest:
    if raw.get("version") != 1:
        raise ManifestError("Secret manifest version must be 1")
    raw_envs = raw.get("environments")
    raw_targets = raw.get("targets")
    raw_secrets = raw.get("secrets", {})
    if not isinstance(raw_envs, dict) or not raw_envs:
        raise ManifestError("At least one environment is required")
    if not isinstance(raw_targets, dict) or not raw_targets:
        raise ManifestError("At least one target is required")
    if not isinstance(raw_secrets, dict):
        raise ManifestError("secrets must be a table")
    environments = {name: _environment(name, value) for name, value in raw_envs.items()}
    targets = {name: _target(name, value) for name, value in raw_targets.items()}
    resources = [(t.platform, t.resource) for t in targets.values() if t.remote]
    if len(resources) != len(set(resources)):
        raise ManifestError("Two targets point at the same remote resource")
    secrets = {
        name: _secret(name, value, environments, targets) for name, value in raw_secrets.items()
    }
    manifest = Manifest(environments, targets, secrets)
    for env in environments:
        for target in targets:
            seen: dict[str, str] = {}
            for spec in secrets.values():
                if env not in spec.envs:
                    continue
                for variable in spec.deliver.get(target, ()):
                    if variable in seen:
                        raise ManifestError(
                            f"{target} receives {variable} from both {seen[variable]} "
                            f"and {spec.name} in {env}"
                        )
                    seen[variable] = spec.name
    return manifest


def load_manifest(path: Path) -> Manifest:
    with path.open("rb") as stream:
        return parse_manifest(tomllib.load(stream))
