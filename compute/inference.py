"""Elastic GPU embedding/reranking tier on Modal (Text Embeddings Inference).

The service's OpenAI-compatible embedding client and TEI rerank client talk to these
endpoints unchanged. Bulk ingestion uses compute/ingestion.py to colocate embedding
and database writes instead. The warm CPU query tier needs measured INT8 parity;
``parity`` quantifies drift and does not automatically authorize index sharing.

Deploy::

    ledgerctl push dev --target modal-tei   # TEI_API_KEY into ledger-tei
    uv run --group compute modal deploy compute/inference.py
    uv run --group compute modal run compute/inference.py::space_id  # LEDGER_EMBEDDING_SPACE_ID

Hub models are pinned to immutable 40-character commits (``LEDGER_EMBED_REVISION``,
``LEDGER_RERANK_REVISION``); branch or tag names are rejected at import so a moved
``main`` can never silently change the served weights. A fine-tuned model on the
``ledger-models`` volume (``LEDGER_EMBED_MODEL=/models/...``) takes no revision and
has no derived space ID: label such a space explicitly after a compatibility check.

Endpoints: ``https://<workspace>--ledger-inference-embed.modal.run/v1/embeddings`` and
``https://<workspace>--ledger-inference-rerank.modal.run/rerank`` (Bearer TEI_API_KEY).
"""

from __future__ import annotations

import os
import re
import subprocess

import modal

APP_NAME = "ledger-inference"
# L4 and L40S are both Ada (sm_89) and share the TEI 89-* image; L4 is cheapest for a
# ~150M encoder, L40S is the fallback when L4 capacity is short.
GPUS = ["L4", "L40S"]
# Keep TEI_VERSION and the space-ID format identical to ledger_rag.ingestion (tested);
# this module cannot import ledger_rag because the serving image does not ship it.
TEI_VERSION = "1.8"
TEI_IMAGE = f"ghcr.io/huggingface/text-embeddings-inference:89-{TEI_VERSION}"
DEFAULT_EMBED = ("Alibaba-NLP/gte-modernbert-base", "e7f32e3c00f91d699e8c43b53106206bcc72bb22")
DEFAULT_RERANK = (
    "Alibaba-NLP/gte-reranker-modernbert-base",
    "f7481e6055501a30fb19d090657df9ec1f79ab2c",
)
_COMMIT = re.compile(r"^[a-f0-9]{40}$")
_HUB_ID = re.compile(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$")


def _pinned(model_var: str, revision_var: str, default: tuple[str, str]) -> tuple[str, str]:
    model = os.environ.get(model_var, default[0])
    revision = os.environ.get(revision_var, default[1] if model == default[0] else "")
    if model.startswith("/models/") and ".." not in model.split("/"):
        if revision:
            raise ValueError(f"{revision_var} must be empty for volume model {model_var}")
    elif not _HUB_ID.fullmatch(model):
        raise ValueError(f"{model_var} must be a Hub ID or a /models/ volume path")
    elif not _COMMIT.fullmatch(revision):
        raise ValueError(f"{revision_var} must be an immutable 40-character commit SHA")
    return model, revision


EMBED_MODEL, EMBED_REVISION = _pinned("LEDGER_EMBED_MODEL", "LEDGER_EMBED_REVISION", DEFAULT_EMBED)
RERANK_MODEL, RERANK_REVISION = _pinned(
    "LEDGER_RERANK_MODEL", "LEDGER_RERANK_REVISION", DEFAULT_RERANK
)
EMBEDDING_SPACE_ID = (
    f"tei-{TEI_VERSION}-float16:{EMBED_MODEL}@{EMBED_REVISION}" if EMBED_REVISION else ""
)
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
            "LEDGER_RERANK_REVISION": RERANK_REVISION,
        }
    )
)
app = modal.App(APP_NAME, image=image)
models_volume = modal.Volume.from_name("ledger-models", create_if_missing=True)
cache_volume = modal.Volume.from_name("ledger-hf-cache", create_if_missing=True)
SECRETS = [modal.Secret.from_name("ledger-tei", required_keys=["TEI_API_KEY"])]


def _router(model_id: str, max_batch_tokens: int, revision: str) -> None:
    subprocess.Popen(
        [
            "text-embeddings-router",
            "--model-id",
            model_id,
            *(["--revision", revision] if revision else []),
            "--dtype",
            "float16",
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
    _router(RERANK_MODEL, 32768, RERANK_REVISION)


@app.local_entrypoint()
def space_id() -> None:
    """Print the LEDGER_EMBEDDING_SPACE_ID matching the served embedding weights."""
    if not EMBEDDING_SPACE_ID:
        raise SystemExit("volume models have no derived space ID; label one explicitly")
    print(EMBEDDING_SPACE_ID)


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
