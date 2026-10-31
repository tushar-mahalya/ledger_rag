"""Single-container GPU embedding and direct Qdrant ingestion; no local model compute."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import modal

from ledger_rag.ingestion.bulk import IngestSpec
from ledger_rag.ingestion.tei import TEI_IMAGE

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data")
image = (
    modal.Image.from_registry(TEI_IMAGE, add_python="3.12")
    .entrypoint([])
    .apt_install("curl")
    .uv_sync(uv_project_dir=str(ROOT), extra_options="--no-dev")
    .env(
        {
            "HUGGINGFACE_HUB_CACHE": "/cache/tei",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "NO_PROXY": "127.0.0.1,localhost",
        }
    )
    .add_local_file(ROOT / "scripts/fetch_finqa.sh", "/opt/ledger/fetch_finqa.sh")
    .add_local_python_source("ledger_rag")
)
app = modal.App("ledger-ingestion", image=image)
data_volume = modal.Volume.from_name("ledger-data", create_if_missing=True)
cache_volume = modal.Volume.from_name("ledger-hf-cache", create_if_missing=True)
database_secret = modal.Secret.from_name(
    "ledger-ingestion", required_keys=["LEDGER_QDRANT_URL", "LEDGER_QDRANT_API_KEY"]
)


def _snapshot_jsonl(input_path: str, target: Path) -> None:
    source = Path(input_path).resolve()
    # Modal mounts Volumes through a symlink, so compare fully resolved paths.
    if not source.is_relative_to(DATA.resolve()) or not source.is_file():
        raise ValueError("JSONL source must be a file inside the data volume")
    shutil.copyfile(source, target)


def _write_report(output: Path, report: dict[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(output)


@app.function(
    gpu=["L4", "L40S"],
    cpu=4,
    memory=16384,
    volumes={"/data": data_volume, "/cache": cache_volume},
    secrets=[database_secret],
    timeout=3600,
    max_containers=1,
    single_use_containers=True,
    retries=0,
)
async def ingest(request: dict[str, object], git_sha: str) -> dict[str, object]:
    import os
    import secrets
    import tempfile
    from dataclasses import asdict
    from urllib.parse import urlsplit
    from uuid import uuid4

    import httpx
    import structlog
    from pydantic import SecretStr
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

    from ledger_rag.core.config import Settings
    from ledger_rag.evaluation.benchmarks.finqa import DATASET_COMMIT, load
    from ledger_rag.evaluation.datasets.fetch import sha256_file
    from ledger_rag.ingestion.bulk import ingest_documents, jsonl_documents
    from ledger_rag.ingestion.tei import TEI_URL, embedding_server
    from ledger_rag.observability.telemetry import configure_telemetry
    from ledger_rag.retrieval.qdrant import QdrantRetriever

    spec = IngestSpec.model_validate(request)
    url = os.environ["LEDGER_QDRANT_URL"]
    key = os.environ["LEDGER_QDRANT_API_KEY"]
    if urlsplit(url).scheme != "https" or not key.strip():
        raise ValueError("Cloud ingestion requires an HTTPS Qdrant URL and a nonempty API key")
    local_key = secrets.token_urlsafe(32)
    settings = Settings(
        demo_token=SecretStr("offline-ingestion-no-api"),
        retrieval_backend="qdrant",
        qdrant_url=url,
        qdrant_api_key=SecretStr(key),
        qdrant_collection=spec.collection,
        qdrant_quantization=spec.quantization,
        qdrant_upsert_batch_size=spec.batch_size,
        embedding_base_url=f"{TEI_URL}/v1",
        embedding_api_key=SecretStr(local_key),
        embedding_model=spec.model_id,
        embedding_space_id=spec.embedding_space_id,
        embedding_dimensions=spec.dimensions,
        embedding_batch_size=spec.batch_size,
        llm_timeout_seconds=120,
        query_deadline_seconds=120,
    )
    configure_telemetry(settings)
    run_id = uuid4().hex
    log = structlog.get_logger(__name__).bind(run_id=run_id, collection=spec.collection)
    store = QdrantRetriever(settings)
    log.info("ingestion.started", embedding_space_id=spec.embedding_space_id)
    input_sha256: str | None = None
    try:
        await data_volume.reload.aio()  # type: ignore[no-untyped-call]
        with tempfile.TemporaryDirectory(prefix="ledger-ingest-") as directory:
            snapshot = Path(directory)
            if spec.source == "finqa":
                fetch = await asyncio.create_subprocess_exec(
                    "bash", "/opt/ledger/fetch_finqa.sh", str(snapshot)
                )
                if await fetch.wait() != 0:
                    raise RuntimeError("Pinned FinQA fetch or checksum verification failed")
                corpus, _ = await asyncio.to_thread(load, snapshot)
                documents = iter(corpus.documents.values())
            else:
                target = snapshot / "documents.jsonl"
                await asyncio.to_thread(_snapshot_jsonl, spec.input_path, target)
                input_sha256 = await asyncio.to_thread(sha256_file, target)
                documents = jsonl_documents(target)
            await store.create_empty_collection()
            async with embedding_server(spec, local_key):
                result = await ingest_documents(
                    documents, store, tenant_id=spec.tenant_id, batch_size=spec.batch_size
                )
        report: dict[str, object] = {
            **asdict(result),
            "run_id": run_id,
            "git_sha": git_sha,
            "collection": spec.collection,
            "embedding_model": spec.model_id,
            "embedding_revision": spec.model_revision,
            "embedding_dimensions": spec.dimensions,
            "embedding_space_id": spec.embedding_space_id,
            "tei_image": TEI_IMAGE,
            "source": spec.source,
            "dataset_commit": DATASET_COMMIT if spec.source == "finqa" else None,
            "input_path": spec.input_path or None,
            "input_sha256": input_sha256,
            "quantization": spec.quantization,
            "status": "verified",
        }
        output = DATA / "ingestion" / f"{run_id}.json"
        report["report_path"] = str(output)
        await asyncio.to_thread(_write_report, output, report)
        await data_volume.commit.aio()  # type: ignore[no-untyped-call]
        log.info("ingestion.report_committed", chunks=result.chunks, report_path=str(output))
        return report
    except (
        RuntimeError,
        ValueError,
        OSError,
        httpx.HTTPError,
        UnexpectedResponse,
        ResponseHandlingException,
    ) as exc:
        log.error("ingestion.failed", error_type=type(exc).__name__)
        raise RuntimeError(
            f"Ingestion {run_id} failed ({type(exc).__name__}); "
            "inspect the isolated collection before any release or retry"
        ) from None
    finally:
        await store.close()


@app.local_entrypoint()
def main(
    collection: str,
    model_revision: str,
    source: str = "finqa",
    input_path: str = "",
    tenant_id: str = "finqa",
    model_id: str = "Alibaba-NLP/gte-modernbert-base",
    dimensions: int = 768,
    batch_size: int = 128,
    quantization: str = "none",
) -> None:
    spec = IngestSpec.model_validate(
        {
            "collection": collection,
            "model_revision": model_revision,
            "source": source,
            "input_path": input_path,
            "tenant_id": tenant_id,
            "model_id": model_id,
            "dimensions": dimensions,
            "batch_size": batch_size,
            "quantization": quantization,
        }
    )
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        git_sha += "-dirty"
    report = ingest.remote(spec.model_dump(), git_sha)
    print(json.dumps(report, indent=2))
