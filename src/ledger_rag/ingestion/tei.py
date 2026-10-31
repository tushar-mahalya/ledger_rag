"""Private TEI process lifetime for a colocated GPU ingestion job."""

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from ledger_rag.ingestion.bulk import IngestSpec

TEI_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:89-1.8"
TEI_URL = "http://127.0.0.1:8000"


@asynccontextmanager
async def embedding_server(
    spec: IngestSpec, api_key: str, *, startup_timeout: float = 600
) -> AsyncIterator[None]:
    if startup_timeout <= 0:
        raise ValueError("TEI startup timeout must be positive")
    if not api_key:
        raise ValueError("TEI requires a nonempty per-job key")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"LEDGER_QDRANT_URL", "LEDGER_QDRANT_API_KEY"}
    }
    environment.update(API_KEY=api_key, AUTO_TRUNCATE="false")
    process = await asyncio.create_subprocess_exec(
        "text-embeddings-router",
        "--model-id",
        spec.model_id,
        "--revision",
        spec.model_revision,
        "--dtype",
        "float16",
        "--hostname",
        "127.0.0.1",
        "--port",
        "8000",
        "--max-client-batch-size",
        str(spec.batch_size),
        "--max-batch-tokens",
        "32768",
        "--json-output",
        env=environment,
    )
    try:
        deadline = asyncio.get_running_loop().time() + startup_timeout
        async with httpx.AsyncClient(
            timeout=2, trust_env=False, headers={"Authorization": f"Bearer {api_key}"}
        ) as client:
            while True:
                if process.returncode is not None:
                    raise RuntimeError(f"TEI exited before readiness (code {process.returncode})")
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("TEI startup deadline exceeded")
                try:
                    response = await client.get(f"{TEI_URL}/health")
                except httpx.TransportError:
                    # The socket is unavailable while the pinned model loads.
                    await asyncio.sleep(0.25)
                    continue
                if response.status_code == 200:
                    break
                if response.status_code != 503:
                    raise RuntimeError(f"TEI readiness returned HTTP {response.status_code}")
                await asyncio.sleep(0.25)
        yield
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except TimeoutError:
                process.kill()
                await process.wait()
