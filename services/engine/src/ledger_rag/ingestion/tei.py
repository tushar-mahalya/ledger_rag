"""Private TEI process lifetime for a colocated GPU ingestion job."""

import asyncio
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Protocol, TextIO

import httpx

TEI_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:89-1.8"
TEI_URL = "http://127.0.0.1:8000"
REDACTED = "[REDACTED]"
_LINE_LIMIT = 1 << 20


class EmbeddingModel(Protocol):
    """What the private TEI process needs (an ``IngestSpec`` or a corpus build)."""

    @property
    def model_id(self) -> str: ...

    @property
    def model_revision(self) -> str: ...

    @property
    def batch_size(self) -> int: ...


@dataclass(frozen=True)
class TeiModel:
    model_id: str
    model_revision: str
    batch_size: int = 128


async def _forward_redacted(stream: asyncio.StreamReader | None, sink: TextIO, secret: str) -> None:
    """Copy router output to the job log with the per-job key removed.

    TEI logs its parsed ``Args`` at startup, including ``api_key: Some(...)``,
    so its output must never reach the Modal log unfiltered.
    """
    if stream is None:
        return
    while True:
        try:
            line = await stream.readline()
        except ValueError:
            line = await stream.read(_LINE_LIMIT)
        if not line:
            return
        sink.write(line.decode(errors="replace").replace(secret, REDACTED))
        sink.flush()


@asynccontextmanager
async def embedding_server(
    spec: EmbeddingModel, api_key: str, *, startup_timeout: float = 600
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
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=_LINE_LIMIT,
    )
    forwarders = [
        asyncio.create_task(_forward_redacted(process.stdout, sys.stdout, api_key)),
        asyncio.create_task(_forward_redacted(process.stderr, sys.stderr, api_key)),
    ]
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
        _, pending = await asyncio.wait(forwarders, timeout=5)
        for task in pending:
            task.cancel()
