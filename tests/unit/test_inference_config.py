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
        "LEDGER_RERANK_REVISION": "b" * 40,
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
    with pytest.warns(UserWarning, match="executing locally"):
        remote.rerank.local()
    argv = popen.call_args.args[0]
    assert argv[argv.index("--model-id") + 1] == expected["LEDGER_RERANK_MODEL"]
    assert argv[argv.index("--revision") + 1] == expected["LEDGER_RERANK_REVISION"]
    assert argv[argv.index("--dtype") + 1] == "float16"


MODEL_VARIABLES = (
    "LEDGER_EMBED_MODEL",
    "LEDGER_EMBED_REVISION",
    "LEDGER_RERANK_MODEL",
    "LEDGER_RERANK_REVISION",
)


def test_defaults_are_pinned_and_space_id_matches_ingestion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("modal", reason="Modal configuration tests require compute group")
    from ledger_rag.ingestion.bulk import IngestSpec
    from ledger_rag.ingestion.tei import TEI_IMAGE

    for key in MODEL_VARIABLES:
        monkeypatch.delenv(key, raising=False)
    module = load_inference("default_inference_config")
    assert module.EMBED_REVISION == "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
    assert module.RERANK_REVISION == "f7481e6055501a30fb19d090657df9ec1f79ab2c"
    assert module.TEI_IMAGE == TEI_IMAGE
    spec = IngestSpec(
        collection="c", model_id=module.EMBED_MODEL, model_revision=module.EMBED_REVISION
    )
    assert module.EMBEDDING_SPACE_ID == spec.embedding_space_id
    assert module.EMBEDDING_SPACE_ID == (
        "tei-1.8-float16:Alibaba-NLP/gte-modernbert-base@e7f32e3c00f91d699e8c43b53106206bcc72bb22"
    )


@pytest.mark.parametrize(
    "environment",
    [
        {"LEDGER_EMBED_REVISION": "main"},
        {"LEDGER_EMBED_REVISION": "v1.0"},
        {"LEDGER_RERANK_REVISION": "main"},
        {"LEDGER_RERANK_REVISION": ""},
        {"LEDGER_EMBED_MODEL": "organization/unpinned"},
        {"LEDGER_EMBED_MODEL": "/models/encoder", "LEDGER_EMBED_REVISION": "a" * 40},
        {"LEDGER_EMBED_MODEL": "../escape"},
    ],
)
def test_mutable_or_invalid_revisions_are_rejected(
    monkeypatch: pytest.MonkeyPatch, environment: dict[str, str]
) -> None:
    pytest.importorskip("modal", reason="Modal configuration tests require compute group")
    for key in MODEL_VARIABLES:
        monkeypatch.delenv(key, raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="LEDGER_"):
        load_inference("invalid_inference_config")


def test_volume_model_is_unrevisioned_without_space_id(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("modal", reason="Modal configuration tests require compute group")
    for key in MODEL_VARIABLES:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LEDGER_EMBED_MODEL", "/models/ledger-embedder/v0001")
    module = load_inference("volume_inference_config")
    assert module.EMBED_REVISION == "" and module.EMBEDDING_SPACE_ID == ""
