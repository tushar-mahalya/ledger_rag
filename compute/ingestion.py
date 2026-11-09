"""Single-container GPU embedding and direct Qdrant ingestion; no local model compute.

    modal run compute/ingestion.py::corpus --name finqa [--promote]   # corpus pipeline

The legacy ``main``/``ingest`` path remains until ``build_corpus`` has run live.
"""

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
    # Hash-pinned export of uv.lock (scripts/export_requirements.py); uv_sync does not
    # support uv workspaces.
    .uv_pip_install(requirements=[str(Path(__file__).with_name("requirements") / "engine.txt")])
    .env(
        {
            "HUGGINGFACE_HUB_CACHE": "/cache/tei",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "NO_PROXY": "127.0.0.1,localhost",
        }
    )
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
    from ledger_rag.evaluation.datasets.fetch import fetch_verified, sha256_file
    from ledger_rag.ingestion.bulk import ingest_documents, jsonl_documents
    from ledger_rag.ingestion.context import SUMMARY_VERSION
    from ledger_rag.ingestion.tei import TEI_URL, embedding_server
    from ledger_rag.observability.telemetry import configure_telemetry
    from ledger_rag.retrieval.qdrant import QdrantRetriever

    spec = IngestSpec.model_validate(request)
    url = os.environ["LEDGER_QDRANT_URL"]
    key = os.environ["LEDGER_QDRANT_API_KEY"]
    if urlsplit(url).scheme != "https" or not key.strip():
        raise ValueError("Cloud ingestion requires an HTTPS Qdrant URL and a nonempty API key")
    local_key = secrets.token_urlsafe(32)
    summary_version = SUMMARY_VERSION if spec.chunk_profile == "v3" else ""
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
        chunk_profile=spec.chunk_profile,
        chunk_summary_version=summary_version,
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
                # Pinned, SHA-256-verified (the shared dataset registry).
                await asyncio.to_thread(fetch_verified, "finqa", snapshot)
                corpus, _ = await asyncio.to_thread(load, snapshot)
                documents = iter(corpus.documents.values())
            else:
                target = snapshot / "documents.jsonl"
                await asyncio.to_thread(_snapshot_jsonl, spec.input_path, target)
                input_sha256 = await asyncio.to_thread(sha256_file, target)
                documents = jsonl_documents(target)
            summaries = None
            if spec.chunk_profile == "v3":
                from ledger_rag.ingestion.context import load_summaries, summary_cache_dir

                summaries = await asyncio.to_thread(
                    load_summaries, summary_cache_dir(DATA / "summaries")
                )
            await store.create_empty_collection()
            async with embedding_server(spec, local_key):
                result = await ingest_documents(
                    documents,
                    store,
                    tenant_id=spec.tenant_id,
                    batch_size=spec.batch_size,
                    chunk_profile=spec.chunk_profile,
                    summaries=summaries,
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
            "chunk_profile": spec.chunk_profile,
            "chunk_summary_version": summary_version or None,
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
async def build_corpus(
    spec_data: dict[str, object], build_id: str, promote: bool, git_sha: str
) -> dict[str, object]:
    """Build any corpus spec with the shared pipeline (ADR 0007) and a private TEI.

    Dataset sources are fetched and verified inside the container; JSONL sources are
    read from the ``ledger-data`` volume (paths relative to /data). The build record
    is stored in the new collection's metadata and as a report on the volume.
    """
    import os
    import secrets
    import tempfile
    from urllib.parse import urlsplit

    from pydantic import SecretStr
    from qdrant_client import AsyncQdrantClient

    from ledger_rag.core.config import Settings
    from ledger_rag.corpus.index import IndexRegistry
    from ledger_rag.corpus.pipeline import BuildRejected, build, plan, store_settings
    from ledger_rag.corpus.sources import load_documents
    from ledger_rag.corpus.spec import CorpusSpec
    from ledger_rag.ingestion import tei
    from ledger_rag.ingestion.context import SUMMARY_VERSION
    from ledger_rag.observability.telemetry import configure_telemetry
    from ledger_rag.retrieval import qdrant

    corpus = CorpusSpec.model_validate(spec_data)
    url = os.environ["LEDGER_QDRANT_URL"]
    key = os.environ["LEDGER_QDRANT_API_KEY"]
    if urlsplit(url).scheme != "https" or not key.strip():
        raise ValueError("Cloud ingestion requires an HTTPS Qdrant URL and a nonempty API key")
    local_key = secrets.token_urlsafe(32)
    v3 = corpus.processing.chunk_profile == "v3"
    base = Settings(
        demo_token=SecretStr("offline-ingestion-no-api"),
        qdrant_url=url,
        qdrant_api_key=SecretStr(key),
        embedding_base_url=f"{tei.TEI_URL}/v1",
        embedding_api_key=SecretStr(local_key),
        embedding_batch_size=128,
        qdrant_upsert_batch_size=128,
        chunk_summary_version=SUMMARY_VERSION if v3 else "",
        llm_timeout_seconds=120,
        query_deadline_seconds=120,
    )
    settings = store_settings(base, corpus, corpus.build_collection(build_id))
    configure_telemetry(settings)
    store = qdrant.QdrantRetriever(settings)
    client = AsyncQdrantClient(url=url, api_key=key)
    try:
        await data_volume.reload.aio()  # type: ignore[no-untyped-call]
        with tempfile.TemporaryDirectory(prefix="ledger-corpus-") as directory:
            root = DATA if corpus.source.kind == "jsonl" else Path(directory)
            documents = await asyncio.to_thread(lambda: list(load_documents(corpus, root)))
        summaries = None
        if v3:
            from ledger_rag.ingestion.context import load_summaries, summary_cache_dir

            summaries = await asyncio.to_thread(
                load_summaries, summary_cache_dir(DATA / "summaries")
            )
        planned = plan(corpus, documents, summaries=summaries)
        model = tei.TeiModel(corpus.embedding.model, corpus.embedding.revision)
        async with tei.embedding_server(model, local_key):
            record = await build(
                planned,
                store,
                build_id=build_id,
                registry=IndexRegistry(client),
                promote=promote,
                summaries=summaries,
                git_sha=git_sha,
            )
        output = DATA / "corpus" / corpus.id / f"{build_id}.json"
        await asyncio.to_thread(_write_report, output, {**record, "tei_image": TEI_IMAGE})
        await data_volume.commit.aio()  # type: ignore[no-untyped-call]
        return record
    except BuildRejected as exc:
        raise RuntimeError(f"{exc} (gate reports: {json.dumps(exc.reports)[:4000]})") from None
    finally:
        await store.close()
        await client.close()


@app.local_entrypoint()
def corpus(name: str, promote: bool = False, build_id: str = "") -> None:
    """``modal run compute/ingestion.py::corpus --name finqa [--promote]``."""
    from ledger_rag.corpus.spec import load_spec, new_build_id, spec_path

    spec = load_spec(spec_path(name, ROOT))
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        git_sha += "-dirty"
    record = build_corpus.remote(
        spec.model_dump(mode="json"), build_id or new_build_id(), promote, git_sha
    )
    print(json.dumps(record, indent=2, default=str))


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
    chunk_profile: str = "v1",
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
            "chunk_profile": chunk_profile,
        }
    )
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        git_sha += "-dirty"
    report = ingest.remote(spec.model_dump(), git_sha)
    print(json.dumps(report, indent=2))
