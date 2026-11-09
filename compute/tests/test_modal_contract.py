"""Every Modal app agrees with the secret manifest, the lockfile exports and naming rules.

Imports each Modal app definition with ``modal.Secret.from_name`` and
``modal.Volume.from_name`` recorded (no network), then checks that:

* each Modal Secret an app attaches is a `ledgerctl push` target in config/secrets.toml
  and that target delivers every key the app requires;
* each image's requirement export exists and is current with uv.lock;
* apps and volumes use the ``ledger-`` namespace.
"""

import importlib.util
from pathlib import Path
from typing import Any

import modal
import pytest

from ledgerctl.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[2]
APPS = sorted(
    [
        p
        for p in (ROOT / "compute").glob("*.py")
        if p.name not in {"__init__.py", "llm_middleware.py"}
    ]
    + [ROOT / "services/voice/modal_app.py"]
)


@pytest.fixture(scope="module")
def recorded() -> dict[str, Any]:
    secrets: list[tuple[str, str, tuple[str, ...]]] = []
    volumes: list[tuple[str, str]] = []
    apps: list[tuple[str, str]] = []
    patch = pytest.MonkeyPatch()
    secret_from_name, volume_from_name = modal.Secret.from_name, modal.Volume.from_name
    app_init = modal.App.__init__

    def record_secret(name: str, *args: Any, required_keys: Any = (), **kwargs: Any) -> Any:
        secrets.append((current, name, tuple(required_keys)))
        return secret_from_name(name, *args, required_keys=list(required_keys), **kwargs)

    def record_volume(name: str, *args: Any, **kwargs: Any) -> Any:
        volumes.append((current, name))
        return volume_from_name(name, *args, **kwargs)

    def record_app(self: Any, name: str | None = None, *args: Any, **kwargs: Any) -> None:
        apps.append((current, name or ""))
        app_init(self, name, *args, **kwargs)

    patch.setattr(modal.Secret, "from_name", record_secret)
    patch.setattr(modal.Volume, "from_name", record_volume)
    patch.setattr(modal.App, "__init__", record_app)
    try:
        for path in APPS:
            current = path.relative_to(ROOT).as_posix()
            spec = importlib.util.spec_from_file_location(f"modal_contract_{path.stem}", path)
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
    finally:
        patch.undo()
    return {"secrets": secrets, "volumes": volumes, "apps": apps}


def test_every_attached_secret_is_delivered_by_ledgerctl(recorded: dict[str, Any]) -> None:
    manifest = load_manifest(ROOT / "config/secrets.toml")
    targets = {t.resource: name for name, t in manifest.targets.items() if t.platform == "modal"}
    assert recorded["secrets"], "no Modal Secrets recorded"
    for app, secret, required in recorded["secrets"]:
        assert secret in targets, f"{app} attaches {secret!r}, which no manifest target pushes"
        delivered = {
            variable
            for spec in manifest.secrets.values()
            for variable in spec.deliver.get(targets[secret], ())
        }
        missing = set(required) - delivered
        assert not missing, f"{app} requires {sorted(missing)} that {secret} never receives"


def test_apps_and_volumes_share_the_ledger_namespace(recorded: dict[str, Any]) -> None:
    assert {path for path, _ in recorded["apps"]} == {p.relative_to(ROOT).as_posix() for p in APPS}
    for app, name in recorded["apps"] + recorded["volumes"]:
        assert name.startswith("ledger-"), f"{app}: {name!r}"


def test_images_install_current_lockfile_exports() -> None:
    import subprocess
    import sys

    for path in APPS:
        source = path.read_text()
        assert "uv_sync(" not in source, f"{path.name}: uv_sync does not support uv workspaces"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_requirements.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
