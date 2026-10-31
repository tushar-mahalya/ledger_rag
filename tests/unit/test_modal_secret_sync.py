import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("target", ["tei", "ingestion"])
def test_vault_projection_uses_protected_file_not_secret_arguments(
    tmp_path: Path, target: str
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    key = "test-only-secret-" * 3
    vault = bin_dir / "vault"
    vault.write_text(f"#!/bin/sh\nprintf '%s' '{key}'\n")
    vault.chmod(0o700)
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, stat, sys\n"
        "path = pathlib.Path(sys.argv[sys.argv.index('--from-json') + 1])\n"
        "assert stat.S_IMODE(path.stat().st_mode) == 0o600\n"
        "assert 'test-only-secret' not in ' '.join(sys.argv)\n"
        "pathlib.Path(os.environ['CAPTURE']).write_text(json.dumps({\n"
        "    'payload': json.loads(path.read_text()), 'path': str(path), 'argv': sys.argv,\n"
        "}))\n"
        "sys.exit(int(os.environ.get('UV_EXIT', '0')))\n"
    )
    uv.chmod(0o700)
    capture = tmp_path / "capture.json"
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "VAULT_ADDR": "https://vault.example.test",
        "LEDGER_QDRANT_URL": "https://qdrant.example.test",
        "CAPTURE": str(capture),
    }
    for exit_code in (0, 1):
        run = subprocess.run(
            ["bash", "-x", str(ROOT / "scripts/sync_modal_secrets.sh"), target],
            env={**environment, "UV_EXIT": str(exit_code)},
            capture_output=True,
            text=True,
        )
        assert run.returncode == exit_code
        assert key not in run.stdout + run.stderr
        data = json.loads(capture.read_text())
        assert not Path(data["path"]).exists()
        if target == "tei":
            assert data["payload"] == {"TEI_API_KEY": key}
        else:
            assert data["payload"] == {
                "LEDGER_QDRANT_URL": environment["LEDGER_QDRANT_URL"],
                "LEDGER_QDRANT_API_KEY": key,
            }
        assert f"ledger-{target}" in data["argv"]


def test_secret_projection_rejects_unknown_target() -> None:
    run = subprocess.run(
        ["bash", str(ROOT / "scripts/sync_modal_secrets.sh"), "unknown"],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0
    assert "usage:" in run.stderr
