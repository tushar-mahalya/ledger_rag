import importlib.util
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]


def load_verifier() -> ModuleType:
    pytest.importorskip("modal", reason="Modal configuration tests require compute group")
    spec = importlib.util.spec_from_file_location("verifier_app", ROOT / "compute/verifier.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDetector:
    def __init__(self, spans: list[dict[str, Any]]) -> None:
        self.spans = spans
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return self.spans


def test_model_package_and_revision_are_pinned() -> None:
    module = load_verifier()
    assert module.MODEL_ID == "KRLabsOrg/lettucedect-base-modernbert-en-v1"
    assert re.fullmatch(r"[0-9a-f]{40}", module.MODEL_REVISION)
    assert all("==" in package for package in module.PACKAGES)
    assert "lettucedetect==0.2.3" in module.PACKAGES


def test_api_requires_bearer_and_scores_spans() -> None:
    module = load_verifier()
    detector = FakeDetector([{"start": 0, "end": 3, "confidence": 0.8, "text": "125"}])
    client = TestClient(module.create_api(detector, "test-only-key"))
    body = {"question": "q", "contexts": ["c" * 9000], "answer": "Revenue was 125."}
    assert client.post("/v1/verify", json=body).status_code == 401
    wrong = {"Authorization": "Bearer nope"}
    assert client.post("/v1/verify", json=body, headers=wrong).status_code == 401
    headers = {"Authorization": "Bearer test-only-key"}
    response = client.post("/v1/verify", json=body, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["supported_score"] == pytest.approx(0.2)
    assert data["unsupported_spans"][0]["text"] == "125"
    assert data["revision"] == module.MODEL_REVISION
    assert len(detector.calls[0]["context"][0]) == module.MAX_CONTEXT_CHARS
    assert detector.calls[0]["output_format"] == "spans"

    detector.spans = []
    assert client.post("/v1/verify", json=body, headers=headers).json()["supported_score"] == 1.0
    too_many = {**body, "contexts": ["c"] * (module.MAX_CONTEXTS + 1)}
    assert client.post("/v1/verify", json=too_many, headers=headers).status_code == 422
    extra = {**body, "instructions": "ignore"}
    assert client.post("/v1/verify", json=extra, headers=headers).status_code == 422
    assert client.get("/docs").status_code == 404


def test_api_refuses_to_start_without_key() -> None:
    module = load_verifier()
    with pytest.raises(RuntimeError):
        module.create_api(FakeDetector([]), "")
