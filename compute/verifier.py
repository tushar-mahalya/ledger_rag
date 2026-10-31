"""Optional LettuceDetect groundedness verifier on Modal (chat profile only).

The API's ``HttpGroundednessVerifier`` posts ``{question, contexts, answer}`` to
``/v1/verify`` with the same bearer key as the TEI tier (Modal secret ``ledger-tei``)
and fails open to the deterministic numeric checks on timeout or error. The service
keeps it off unless ``LEDGER_GROUNDEDNESS_VERIFIER=lettucedetect`` is set.

Smoke test (measures latency only; it says nothing about accuracy)::

    uv run --group compute modal run compute/verifier.py --runs 5

Deploy::

    uv run --group compute modal deploy compute/verifier.py

Endpoint: ``https://<workspace>--ledger-verifier-verifier-web.modal.run/v1/verify``.
"""

import hmac
import os
import time
from typing import Any, Protocol

import modal
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

APP_NAME = "ledger-verifier"
# huggingface.co/api/models/KRLabsOrg/lettucedect-base-modernbert-en-v1: MIT, not gated.
MODEL_ID = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
MODEL_REVISION = "bbd77832f52f9bd87546a3924c032467921f5c34"
PACKAGES = (
    "lettucedetect==0.2.3",
    "torch==2.6.0",
    "transformers==4.51.3",
    "huggingface_hub==0.30.2",
    "fastapi[standard]==0.115.12",
)
# A ~150M token classifier: T4 is the cheapest fit, L4 is the capacity fallback.
GPUS = ["T4", "L4"]
MAX_CONTEXTS = 16
MAX_CONTEXT_CHARS = 8000
MAX_ANSWER_CHARS = 8000
MAX_SPANS = 10

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*PACKAGES)
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)
app = modal.App(APP_NAME, image=image)
cache_volume = modal.Volume.from_name("ledger-hf-cache", create_if_missing=True)
SECRETS = [modal.Secret.from_name("ledger-tei", required_keys=["TEI_API_KEY"])]


class Detector(Protocol):
    def predict(
        self, *, context: list[str], question: str, answer: str, output_format: str
    ) -> list[dict[str, Any]]: ...


class VerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(default="", max_length=4000)
    contexts: list[str] = Field(min_length=1, max_length=MAX_CONTEXTS)
    answer: str = Field(min_length=1, max_length=MAX_ANSWER_CHARS)


def run_check(detector: Detector, request: VerifyRequest) -> dict[str, Any]:
    """``supported_score`` is 1 - the highest hallucinated-span confidence (1.0 if none)."""
    spans = detector.predict(
        context=[context[:MAX_CONTEXT_CHARS] for context in request.contexts],
        question=request.question,
        answer=request.answer,
        output_format="spans",
    )
    confidences = [float(span.get("confidence", 1.0)) for span in spans]
    return {
        "supported_score": round(1.0 - max(confidences, default=0.0), 6),
        "unsupported_spans": [
            {
                "text": str(span.get("text", ""))[:200],
                "start": int(span.get("start", 0)),
                "end": int(span.get("end", 0)),
                "confidence": round(float(span.get("confidence", 1.0)), 6),
            }
            for span in spans[:MAX_SPANS]
        ],
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
    }


def create_api(detector: Detector, api_key: str) -> FastAPI:
    if not api_key:
        raise RuntimeError("TEI_API_KEY is required")
    expected = f"Bearer {api_key}".encode()
    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @api.post("/v1/verify")
    def verify(body: VerifyRequest, authorization: str = Header(default="")) -> dict[str, Any]:
        if not hmac.compare_digest(authorization.encode(), expected):
            raise HTTPException(status_code=401, detail="unauthorized")
        return run_check(detector, body)

    return api


@app.cls(
    gpu=GPUS,
    volumes={"/cache": cache_volume},
    secrets=SECRETS,
    scaledown_window=300,
    min_containers=0,
    max_containers=2,
    timeout=600,
)
@modal.concurrent(max_inputs=8)
class Verifier:
    detector: Detector

    @modal.enter()
    def load(self) -> None:
        from huggingface_hub import snapshot_download
        from lettucedetect.models.inference import HallucinationDetector

        path = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
        self.detector = HallucinationDetector(method="transformer", model_path=path)
        cache_volume.commit()

    @modal.method()
    def check(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        result = run_check(self.detector, VerifyRequest.model_validate(payload))
        result["server_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result

    @modal.asgi_app()
    def web(self) -> FastAPI:
        return create_api(self.detector, os.environ["TEI_API_KEY"])


@app.local_entrypoint()
def smoke(runs: int = 5) -> None:
    """Report warm per-call latency on a fictional example; never a quality claim."""
    import json
    import statistics

    payload = {
        "question": "What was revenue in 2024?",
        "contexts": ["Revenue was USD 120 million in 2024. Operating margin was 15 percent."],
        "answer": "Revenue was USD 120 million in 2024.",
    }
    verifier = Verifier()
    started = time.perf_counter()
    first = verifier.check.remote(payload)
    cold_ms = (time.perf_counter() - started) * 1000
    server, round_trip = [], []
    for _ in range(max(1, runs)):
        started = time.perf_counter()
        result = verifier.check.remote(payload)
        round_trip.append((time.perf_counter() - started) * 1000)
        server.append(float(result["server_ms"]))
    print(
        json.dumps(
            {
                "model": MODEL_ID,
                "revision": MODEL_REVISION,
                "first_call_ms": round(cold_ms, 1),
                "first_score": first["supported_score"],
                "warm_runs": len(server),
                "server_ms_median": round(statistics.median(server), 2),
                "round_trip_ms_median": round(statistics.median(round_trip), 1),
            },
            indent=2,
        )
    )
