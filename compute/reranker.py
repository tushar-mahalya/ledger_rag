"""Qwen3-Reranker on vLLM behind bearer auth (Modal L4/L40S, scale to zero).

vLLM loads the original Qwen3-Reranker checkpoints as sequence classifiers with the
documented conversion (``Qwen3ForSequenceClassification`` scoring the "no"/"yes"
logits) and renders each pair with the official instruction template
(``qwen3_reranker.jinja``, copied verbatim from vLLM v0.32.0
``examples/pooling/score/template``). vLLM listens on loopback only: its non-``/v1``
rerank route ignores ``--api-key``, so a small proxy enforces the ``ledger-tei``
bearer key and exposes a TEI-compatible ``POST /rerank``. The existing
``RerankingRetriever`` therefore works unchanged; ``VllmRerankingRetriever`` adds the
instruction and per-profile routing.

    uv run --locked --group compute modal deploy compute/reranker.py
    uv run --locked --group compute modal run compute/reranker.py::bench   # latency
"""

import hmac
import json
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import modal

if TYPE_CHECKING:
    import httpx
    from fastapi import FastAPI

APP_NAME = "ledger-reranker"
VLLM_VERSION = "0.32.0"
MODELS = {
    "0.6b": ("Qwen/Qwen3-Reranker-0.6B", "e61197ed45024b0ed8a2d74b80b4d909f1255473"),
    "4b": ("Qwen/Qwen3-Reranker-4B", "22e683669bc0f0bd69640a1354a6d0aebcfeede5"),
}
GPU = {"0.6b": "L4", "4b": "L40S"}
HF_OVERRIDES = {
    "architectures": ["Qwen3ForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}
DEFAULT_INSTRUCTION = (
    "Given a question about a company's SEC filing, retrieve the filing passage or table "
    "that contains the facts needed to answer it"
)
SERVED_NAME = "qwen3-reranker"
TEMPLATE = "/opt/ledger/qwen3_reranker.jinja"
VLLM_PORT = 8001
MAX_MODEL_LEN = 8192
MAX_DOC_TOKENS = 6144
MAX_DOCUMENTS = 100
MAX_CHARS = 32_000
SCALEDOWN = int(os.environ.get("LEDGER_RERANK_SCALEDOWN", "300"))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "fastapi", "httpx")
    .env(
        {
            "HF_HUB_CACHE": "/cache/vllm-hf",
            "VLLM_CACHE_ROOT": "/cache/vllm",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "LEDGER_RERANK_SCALEDOWN": str(SCALEDOWN),
        }
    )
    .add_local_file(Path(__file__).with_name("qwen3_reranker.jinja"), TEMPLATE)
)
app = modal.App(APP_NAME, image=image)
cache_volume = modal.Volume.from_name("ledger-hf-cache", create_if_missing=True)
SECRETS = [modal.Secret.from_name("ledger-tei", required_keys=["TEI_API_KEY"])]


def vllm_command(model_key: str) -> list[str]:
    model, revision = MODELS[model_key]
    return [
        "vllm",
        "serve",
        model,
        "--revision",
        revision,
        "--runner",
        "pooling",
        "--hf-overrides",
        json.dumps(HF_OVERRIDES),
        "--chat-template",
        TEMPLATE,
        "--served-model-name",
        SERVED_NAME,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        "0.85",
        "--host",
        "127.0.0.1",
        "--port",
        str(VLLM_PORT),
    ]


def create_proxy(
    model_key: str, api_key: str, upstream: "httpx.AsyncClient | None" = None
) -> "FastAPI":
    """TEI-compatible authenticated front for a loopback vLLM rerank server."""
    import httpx
    from fastapi import FastAPI, HTTPException, Request, Response
    from pydantic import BaseModel, ConfigDict, Field, ValidationError

    if not api_key.strip():
        raise ValueError("A nonempty serving key is required")
    model, revision = MODELS[model_key]
    expected = f"Bearer {api_key}".encode()
    client = upstream or httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{VLLM_PORT}", timeout=120, trust_env=False
    )

    class RerankBody(BaseModel):
        model_config = ConfigDict(extra="ignore")

        query: str = Field(min_length=1, max_length=4000)
        texts: list[str] = Field(min_length=1, max_length=MAX_DOCUMENTS)
        instruction: str | None = Field(default=None, max_length=500)

    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @api.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.post("/rerank")
    async def rerank(request: Request, response: Response) -> list[dict[str, Any]]:
        supplied = request.headers.get("authorization", "").encode()
        if not hmac.compare_digest(supplied, expected):
            raise HTTPException(401, "Unauthorized")
        try:
            body = RerankBody.model_validate(await request.json())
        except (ValueError, ValidationError):
            raise HTTPException(422, "Invalid rerank request") from None
        if any(len(text) > MAX_CHARS for text in body.texts):
            raise HTTPException(413, "Document too long")
        start = time.perf_counter()
        try:
            upstream_response = await client.post(
                "/v1/rerank",
                json={
                    "model": SERVED_NAME,
                    "query": body.query,
                    "documents": body.texts,
                    "instruction": body.instruction or DEFAULT_INSTRUCTION,
                    "max_tokens_per_doc": MAX_DOC_TOKENS,
                },
            )
        except httpx.HTTPError:
            raise HTTPException(502, "Reranker upstream unavailable") from None
        if upstream_response.status_code != 200:
            raise HTTPException(502, f"Reranker upstream HTTP {upstream_response.status_code}")
        try:
            results = [
                {"index": int(item["index"]), "score": float(item["relevance_score"])}
                for item in upstream_response.json()["results"]
            ]
        except (KeyError, TypeError, ValueError):
            raise HTTPException(502, "Invalid reranker upstream response") from None
        if sorted(r["index"] for r in results) != list(range(len(body.texts))):
            raise HTTPException(502, "Reranker upstream did not score every document")
        response.headers["x-upstream-ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
        response.headers["x-reranker-model"] = f"{model}@{revision}"
        return sorted(results, key=lambda r: (-r["score"], r["index"]))

    return api


class _VllmServer:
    model_key = ""

    @modal.enter()
    def start(self) -> None:
        import httpx

        self.process = subprocess.Popen(vllm_command(self.model_key))
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"vLLM exited during startup ({self.process.returncode})")
            try:
                if httpx.get(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=2).is_success:
                    return
            except httpx.TransportError:
                pass
            time.sleep(1)
        raise TimeoutError("vLLM startup deadline exceeded")

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()

    @modal.asgi_app()
    def web(self) -> "FastAPI":
        return create_proxy(self.model_key, os.environ["TEI_API_KEY"])


COMMON: dict[str, Any] = {
    "volumes": {"/cache": cache_volume},
    "secrets": SECRETS,
    "scaledown_window": SCALEDOWN,
    "min_containers": 0,
    "max_containers": 2,
    "timeout": 3600,
    "startup_timeout": 900,
}


@app.cls(gpu=GPU["0.6b"], **COMMON)
@modal.concurrent(max_inputs=32)
class Qwen3Reranker06B(_VllmServer):
    model_key = "0.6b"


@app.cls(gpu=GPU["4b"], **COMMON)
@modal.concurrent(max_inputs=32)
class Qwen3Reranker4B(_VllmServer):
    model_key = "4b"


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(fraction * len(ordered)))], 1)


@app.function(
    secrets=SECRETS, cpu=1, timeout=1800, image=modal.Image.debian_slim().pip_install("httpx")
)
def measure(
    url: str, queries: list[str], documents: list[str], sizes: list[int], samples: int
) -> dict[str, Any]:
    """Client-observed latency from inside Modal (no local secret handling)."""
    import httpx

    headers = {"Authorization": f"Bearer {os.environ['TEI_API_KEY']}"}
    report: dict[str, Any] = {}
    with httpx.Client(base_url=url, headers=headers, timeout=900) as client:
        start = time.perf_counter()
        client.post(
            "/rerank", json={"query": queries[0], "texts": documents[:2]}
        ).raise_for_status()
        report["cold_start_s"] = round(time.perf_counter() - start, 1)
        unauthorized = client.post(
            "/rerank",
            json={"query": "q", "texts": ["d"]},
            headers={"Authorization": "Bearer wrong"},
        )
        report["unauthorized_status"] = unauthorized.status_code
        for size in sizes:
            totals: list[float] = []
            upstream: list[float] = []
            for index in range(samples + 3):
                query = queries[index % len(queries)]
                offset = (index * 7) % max(1, len(documents) - size)
                batch = documents[offset : offset + size]
                start = time.perf_counter()
                response = client.post("/rerank", json={"query": query, "texts": batch})
                elapsed = (time.perf_counter() - start) * 1000
                response.raise_for_status()
                if len(response.json()) != size:
                    raise RuntimeError("Reranker did not score every candidate")
                if index >= 3:  # warm-up requests are excluded
                    totals.append(elapsed)
                    upstream.append(float(response.headers["x-upstream-ms"]))
            report[f"n{size}"] = {
                "samples": len(totals),
                "p50_ms": _percentile(totals, 0.5),
                "p95_ms": _percentile(totals, 0.95),
                "upstream_p50_ms": _percentile(upstream, 0.5),
                "upstream_p95_ms": _percentile(upstream, 0.95),
            }
        report["model"] = response.headers.get("x-reranker-model")
    return report


@app.local_entrypoint()
def bench(models: str = "0.6b,4b", samples: int = 30, sizes: str = "20,50") -> None:
    """p50/p95 for ~1,800-char FinQA chunks (v1 contextual text) per model and size."""
    from ledger_rag.evaluation.benchmarks.finqa import load
    from ledger_rag.ingestion.chunking import chunk_document, contextual_text

    corpus, cases = load(Path(__file__).resolve().parents[1] / "data/finqa")
    documents = [
        contextual_text(chunk)
        for document in corpus.documents.values()
        for chunk in chunk_document(document)
        if 1650 <= len(chunk.text) <= 1800
    ][:400]
    queries = [case.question for case in cases[:100]]
    size_list = [int(size) for size in sizes.split(",")]
    servers = {"0.6b": Qwen3Reranker06B, "4b": Qwen3Reranker4B}
    results: dict[str, Any] = {
        "document_chars_mean": round(sum(map(len, documents)) / len(documents)),
        "vllm": VLLM_VERSION,
    }
    for key in models.split(","):
        url = servers[key]().web.get_web_url()
        if not url:
            raise SystemExit("Reranker web endpoint has no URL")
        results[key] = {
            "gpu": GPU[key],
            **measure.remote(url, queries, documents, size_list, samples),
        }
        print(json.dumps({key: results[key]}, indent=2), flush=True)
    print(json.dumps(results, indent=2))
