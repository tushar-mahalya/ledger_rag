"""Elastic GPU embedding/reranking tier on Modal (Text Embeddings Inference).

The service's OpenAI-compatible embedding client and TEI rerank client talk to these
endpoints unchanged. Bulk ingestion uses compute/ingestion.py to colocate embedding
and database writes instead. The warm CPU query tier needs measured INT8 parity;
``parity`` quantifies drift and does not automatically authorize index sharing.

Deploy::

    scripts/sync_modal_secrets.sh        # projects Vault's embedding key into Modal
    LEDGER_EMBED_MODEL=/models/ledger-embedder/v0001 \
        uv run --group compute modal deploy compute/inference.py

Endpoints: ``https://<workspace>--ledger-inference-embed.modal.run/v1/embeddings`` and
``https://<workspace>--ledger-inference-rerank.modal.run/rerank`` (Bearer TEI_API_KEY).
"""

from __future__ import annotations

import os
import subprocess

import modal

APP_NAME = "ledger-inference"
# L4 and L40S are both Ada (sm_89) and share the TEI 89-* image; L4 is cheapest for a
# ~150M encoder, L40S is the fallback when L4 capacity is short.
GPUS = ["L4", "L40S"]
TEI_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:89-1.8"
EMBED_MODEL = os.environ.get("LEDGER_EMBED_MODEL", "Alibaba-NLP/gte-modernbert-base")
EMBED_REVISION = os.environ.get("LEDGER_EMBED_REVISION", "")
RERANK_MODEL = os.environ.get("LEDGER_RERANK_MODEL", "Alibaba-NLP/gte-reranker-modernbert-base")
PORT = 8000

image = (
    modal.Image.from_registry(TEI_IMAGE, add_python="3.12")
    .entrypoint([])
    .env(
        {
            "HUGGINGFACE_HUB_CACHE": "/cache/tei",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "LEDGER_EMBED_MODEL": EMBED_MODEL,
            "LEDGER_EMBED_REVISION": EMBED_REVISION,
            "LEDGER_RERANK_MODEL": RERANK_MODEL,
        }
    )
)
app = modal.App(APP_NAME, image=image)
models_volume = modal.Volume.from_name("ledger-models", create_if_missing=True)
cache_volume = modal.Volume.from_name("ledger-hf-cache", create_if_missing=True)
SECRETS = [modal.Secret.from_name("ledger-tei", required_keys=["TEI_API_KEY"])]


def _router(model_id: str, max_batch_tokens: int, revision: str = "") -> None:
    subprocess.Popen(
        [
            "text-embeddings-router",
            "--model-id",
            model_id,
            *(["--revision", revision, "--dtype", "float16"] if revision else []),
            "--port",
            str(PORT),
            "--api-key",
            os.environ["TEI_API_KEY"],
            "--max-client-batch-size",
            "128",
            "--max-batch-tokens",
            str(max_batch_tokens),
            "--auto-truncate",
            "--json-output",
        ]
    )


COMMON = {
    "gpu": GPUS,
    "volumes": {"/models": models_volume, "/cache": cache_volume},
    "secrets": SECRETS,
    "scaledown_window": 300,
    "min_containers": 0,
    "max_containers": 4,
    "timeout": 3600,
}


@app.function(**COMMON)  # type: ignore[arg-type]
@modal.concurrent(max_inputs=128)
@modal.web_server(port=PORT, startup_timeout=600)
def embed() -> None:
    _router(EMBED_MODEL, 65536, EMBED_REVISION)


@app.function(**COMMON)  # type: ignore[arg-type]
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=600)
def rerank() -> None:
    _router(RERANK_MODEL, 32768)


@app.local_entrypoint()
def parity(local_url: str = "http://127.0.0.1:18080", samples: int = 64) -> None:
    """Cosine agreement between Modal GPU (fp16) and local CPU (INT8) embeddings."""
    import json
    import math
    from pathlib import Path

    import httpx

    from ledger_rag.evaluation.benchmarks.finqa import load
    from ledger_rag.ingestion.chunking import chunk_document, contextual_text

    corpus, cases = load(Path("data/finqa"))
    passages = [
        contextual_text(c) for d in list(corpus.documents.values())[:40] for c in chunk_document(d)
    ]
    texts = passages[: samples // 2] + [c.question for c in cases[: samples - samples // 2]]
    key = os.environ["TEI_API_KEY"]
    remote_url = embed.get_web_url()
    if not remote_url:
        raise SystemExit("deploy the app first: modal deploy compute/inference.py")
    payload = {"input": texts, "model": "parity"}

    def call(url: str, auth: str | None) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {auth}"} if auth else {}
        resp = httpx.post(f"{url}/v1/embeddings", json=payload, headers=headers, timeout=300)
        resp.raise_for_status()
        return [row["embedding"] for row in sorted(resp.json()["data"], key=lambda r: r["index"])]

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

    gpu, cpu = call(remote_url, key), call(local_url, None)
    scores = sorted(cosine(a, b) for a, b in zip(gpu, cpu, strict=True))
    print(
        json.dumps(
            {
                "samples": len(scores),
                "min": round(scores[0], 5),
                "p05": round(scores[int(0.05 * len(scores))], 5),
                "mean": round(sum(scores) / len(scores), 5),
            },
            indent=2,
        )
    )
