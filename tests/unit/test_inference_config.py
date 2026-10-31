import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_inference(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "compute/inference.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_and_revision_survive_remote_module_reimport(monkeypatch: pytest.MonkeyPatch) -> None:
    modal = pytest.importorskip("modal", reason="Modal configuration tests require compute group")
    captured: dict[str, str] = {}
    original_env = modal.Image.env

    def record_env(image, values):
        captured.update(values)
        return original_env(image, values)

    monkeypatch.setattr(modal.Image, "env", record_env)
    expected = {
        "LEDGER_EMBED_MODEL": "organization/pinned-encoder",
        "LEDGER_EMBED_REVISION": "a" * 40,
        "LEDGER_RERANK_MODEL": "organization/pinned-reranker",
    }
    for key, value in expected.items():
        monkeypatch.setenv(key, value)
    load_inference("local_inference_config")
    assert {key: captured.get(key) for key in expected} == expected

    for key in expected:
        monkeypatch.delenv(key)
    for key, value in captured.items():
        monkeypatch.setenv(key, value)
    remote = load_inference("remote_inference_config")
    monkeypatch.setenv("TEI_API_KEY", "test-only-serving-key")
    popen = Mock()
    monkeypatch.setattr(remote.subprocess, "Popen", popen)
    with pytest.warns(UserWarning, match="executing locally"):
        remote.embed.local()
    argv = popen.call_args.args[0]
    assert argv[argv.index("--model-id") + 1] == expected["LEDGER_EMBED_MODEL"]
    assert argv[argv.index("--revision") + 1] == expected["LEDGER_EMBED_REVISION"]
    assert argv[argv.index("--dtype") + 1] == "float16"
