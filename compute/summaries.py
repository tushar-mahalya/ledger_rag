"""Per-document LLM summaries for chunk profile v3, generated on Modal CPU workers.

Summaries are cached on the ``ledger-data`` Volume keyed by document content hash
(``/data/summaries/<prompt version>/<model>/``) and read back by
``compute/ingestion.py --chunk-profile v3``. Running this separately keeps the LLM
key out of the GPU/Qdrant container and avoids paying for an idle GPU while the
rate-limited LLM calls (at most 8 in flight) complete.

    uv run --locked --group compute modal run compute/summaries.py [--limit 20]
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl")
    .uv_sync(uv_project_dir=str(ROOT), extra_options="--no-dev")
    .add_local_file(ROOT / "scripts/fetch_finqa.sh", "/opt/ledger/fetch_finqa.sh")
    .add_local_python_source("ledger_rag")
)
app = modal.App("ledger-summaries", image=image)
data_volume = modal.Volume.from_name("ledger-data", create_if_missing=True)
llm_secret = modal.Secret.from_name(
    "ledger-llm", required_keys=["LEDGER_LLM_BASE_URL", "LEDGER_LLM_API_KEY"]
)


@app.function(
    cpu=2,
    memory=4096,
    volumes={"/data": data_volume},
    secrets=[llm_secret],
    timeout=3600,
    max_containers=1,
    retries=0,
)
async def summarize(model: str, limit: int, git_sha: str) -> dict[str, object]:
    import os
    import tempfile
    import time
    from urllib.parse import urlsplit
    from uuid import uuid4

    import httpx

    from ledger_rag.evaluation.benchmarks.finqa import DATASET_COMMIT, load
    from ledger_rag.ingestion.context import (
        SUMMARY_PROMPT_VERSION,
        load_summaries,
        summarize_documents,
        summary_cache_dir,
    )

    base_url = os.environ["LEDGER_LLM_BASE_URL"].rstrip("/") + "/"
    if urlsplit(base_url).scheme != "https":
        raise ValueError("Summary generation requires an HTTPS LLM endpoint")
    run_id = uuid4().hex
    start = time.monotonic()
    await data_volume.reload.aio()  # type: ignore[no-untyped-call]
    with tempfile.TemporaryDirectory(prefix="ledger-summaries-") as directory:
        fetch = await asyncio.create_subprocess_exec(
            "bash", "/opt/ledger/fetch_finqa.sh", directory
        )
        if await fetch.wait() != 0:
            raise RuntimeError("Pinned FinQA fetch or checksum verification failed")
        corpus, _ = await asyncio.to_thread(load, Path(directory))
    documents = list(corpus.documents.values())
    if limit > 0:
        documents = documents[:limit]
    cache_dir = summary_cache_dir(DATA / "summaries", model)
    commits: list[asyncio.Task[None]] = []

    def checkpoint(generated: int) -> None:
        if generated % 200 == 0:
            commits.append(asyncio.create_task(data_volume.commit.aio()))  # type: ignore[no-untyped-call]

    async with httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": f"Bearer {os.environ['LEDGER_LLM_API_KEY']}"},
        timeout=httpx.Timeout(90, connect=10),
        follow_redirects=False,
    ) as client:
        stats = await summarize_documents(
            documents, cache_dir, client, model=model, on_written=checkpoint
        )
    await asyncio.gather(*commits)
    cached = await asyncio.to_thread(load_summaries, cache_dir)
    report: dict[str, object] = {
        **stats,
        "run_id": run_id,
        "git_sha": git_sha,
        "model": model,
        "prompt_version": SUMMARY_PROMPT_VERSION,
        "documents": len(documents),
        "cache_entries": len(cached),
        "cache_dir": str(cache_dir),
        "dataset_commit": DATASET_COMMIT,
        "seconds": round(time.monotonic() - start, 1),
        "sample": next(iter(cached.values()), None),
    }
    output = DATA / "summaries" / "runs" / f"{run_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    await data_volume.commit.aio()  # type: ignore[no-untyped-call]
    return report


@app.local_entrypoint()
def main(model: str = "gemini-3.5-flash-lite", limit: int = 0) -> None:
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        git_sha += "-dirty"
    report = summarize.remote(model, limit, git_sha)
    print(json.dumps(report, indent=2))
    if report["failed"]:
        raise SystemExit(f"{report['failed']} summaries failed; rerun to fill the cache")
