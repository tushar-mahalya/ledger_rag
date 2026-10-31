"""Read-only FinQA evaluation on Modal: loopback TEI embedder + reranker on one GPU.

The job never writes to Qdrant: it only calls ``count`` and ``search`` (which open the
collection with ``create=False`` and verify its embedding identity). Reports, including
lineage and GPU wall time, are written to ``/data/evaluation/<run_id>.json``.

    uv run --locked --group compute modal run compute/evaluation.py --command all \\
        --limit 20 --answer-samples 20
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

import modal

from ledger_rag.ingestion.tei import TEI_IMAGE

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data")
COLLECTION = "finqa_gte_modernbert_v1"
EMBEDDING_MODEL = "Alibaba-NLP/gte-modernbert-base"
EMBEDDING_REVISION = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
EMBEDDING_SPACE_ID = f"tei-1.8-float16:{EMBEDDING_MODEL}@{EMBEDDING_REVISION}"
RERANKER_MODEL = "Alibaba-NLP/gte-reranker-modernbert-base"
RERANKER_REVISION = "f7481e6055501a30fb19d090657df9ec1f79ab2c"
# Fingerprint and size recorded by ingestion run adb0f1a36082405b8497e2c39d15bfe3.
CORPUS_SHA256 = "9b455b1b21167b4fee7592f9cc2a68e5d12a336473156137b5ef0692e442ae9d"
EXPECTED_CHUNKS = 8611
EMBED_PORT, RERANK_PORT = 8000, 8001
# The TEI ``89-*`` image needs compute capability 8.9: L4 preferred, L40S when L4
# capacity is unavailable. The actual device is recorded in the report.
GPU = ["L4", "L40S"]
GPU_COMMANDS = ("retrieval", "contexts")
COMMANDS = (*GPU_COMMANDS, "answers")
SECRET_ENV = {
    "LEDGER_QDRANT_URL",
    "LEDGER_QDRANT_API_KEY",
    "LEDGER_LLM_BASE_URL",
    "LEDGER_LLM_API_KEY",
}

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
app = modal.App("ledger-evaluation", image=image)
data_volume = modal.Volume.from_name("ledger-data", create_if_missing=True)
cache_volume = modal.Volume.from_name("ledger-hf-cache", create_if_missing=True)
database_secret = modal.Secret.from_name(
    "ledger-ingestion", required_keys=["LEDGER_QDRANT_URL", "LEDGER_QDRANT_API_KEY"]
)
llm_secret = modal.Secret.from_name(
    "ledger-llm", required_keys=["LEDGER_LLM_BASE_URL", "LEDGER_LLM_API_KEY"]
)


def benchmark_argv(
    command: str, options: dict[str, Any], data_dir: str, key: str, contexts: str = ""
) -> list[str]:
    """Translate validated job options into ``finqa`` CLI arguments; endpoints are fixed."""
    if command not in COMMANDS:
        raise ValueError("Evaluation job only runs read-only retrieval/contexts/answers commands")
    argv = [
        command,
        "--data",
        data_dir,
        "--split",
        str(options.get("split", "dev")),
        "--qdrant-url",
        os.environ.get("LEDGER_QDRANT_URL", ""),
        "--collection",
        COLLECTION,
        "--embedding-url",
        f"http://127.0.0.1:{EMBED_PORT}/v1",
        "--embedding-key",
        key,
        "--embedding-model",
        EMBEDDING_MODEL,
        "--embedding-revision",
        EMBEDDING_REVISION,
        "--embedding-space-id",
        EMBEDDING_SPACE_ID,
        "--reranker-url",
        f"http://127.0.0.1:{RERANK_PORT}",
        "--reranker-key",
        key,
        "--reranker-model",
        RERANKER_MODEL,
        "--reranker-revision",
        RERANKER_REVISION,
        "--expected-corpus-sha256",
        CORPUS_SHA256,
        "--limit",
        str(int(options.get("limit", 0))),
        "--configs",
        *options.get("configs", ["sparse", "dense", "hybrid", "hybrid:rerank"]),
        "--scopes",
        *options.get("scopes", ["company", "global"]),
        "--retrieval-concurrency",
        str(int(options.get("retrieval_concurrency", 4))),
        "--llm-models",
        *options.get("models", ["gemini-3.5-flash:low", "gemini-3.5-flash-lite"]),
        "--llm-concurrency",
        str(int(options.get("llm_concurrency", 8))),
        "--temperature",
        str(options.get("temperature", "default")),
        "--max-tokens",
        str(int(options.get("max_tokens", 2048))),
        "--answer-samples",
        str(int(options.get("answer_samples", 200))),
        "--answer-modes",
        *options.get("answer_modes", ["retrieved@3", "retrieved@5", "oracle"]),
        "--seed",
        str(int(options.get("seed", 13))),
        "--git-sha",
        str(options.get("git_sha", "unknown")),
    ]
    if options.get("llm_rpm"):
        argv += ["--llm-rpm", str(float(options["llm_rpm"]))]
    if contexts:
        argv += ["--contexts", contexts]
    if options.get("llm_seed") is not None:
        argv += ["--llm-seed", str(int(options["llm_seed"]))]
    if options.get("unseal_test"):
        argv.append("--unseal-test")
    return argv


@asynccontextmanager
async def tei_server(
    model: str, revision: str, port: int, key: str, *, startup_timeout: float = 900
) -> AsyncIterator[None]:
    """Loopback TEI router; secrets are stripped from its environment."""
    import httpx

    environment = {k: v for k, v in os.environ.items() if k not in SECRET_ENV}
    environment.update(API_KEY=key, AUTO_TRUNCATE="false")
    process = await asyncio.create_subprocess_exec(
        "text-embeddings-router",
        "--model-id",
        model,
        "--revision",
        revision,
        "--dtype",
        "float16",
        "--hostname",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-client-batch-size",
        "64",
        "--max-batch-tokens",
        "32768",
        "--json-output",
        env=environment,
    )
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + startup_timeout
        async with httpx.AsyncClient(
            timeout=2, trust_env=False, headers={"Authorization": f"Bearer {key}"}
        ) as client:
            while True:
                if process.returncode is not None:
                    raise RuntimeError(f"TEI {model} exited early (code {process.returncode})")
                if loop.time() >= deadline:
                    raise TimeoutError(f"TEI {model} startup deadline exceeded")
                try:
                    response = await client.get(f"http://127.0.0.1:{port}/health")
                except httpx.TransportError:
                    await asyncio.sleep(0.5)
                    continue
                if response.status_code == 200:
                    break
                if response.status_code != 503:
                    raise RuntimeError(f"TEI readiness returned HTTP {response.status_code}")
                await asyncio.sleep(0.5)
        yield
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except TimeoutError:
                process.kill()
                await process.wait()


def _gpu_name() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_report(output: Path, report: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(output)


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    """Compact, secret-free summary returned to the local entrypoint."""
    summary: dict[str, Any] = {
        key: report.get(key)
        for key in ("run_id", "report_path", "gpu", "gpu_seconds", "cpu_seconds", "git_sha")
    }
    retrieval = report.get("retrieval", {}).get("retrieval")
    if retrieval:
        summary["retrieval"] = {
            arm: {
                name: metric
                for name, metric in values["metrics"].items()
                if name
                in {
                    "page_recall@1",
                    "page_recall@5",
                    "page_mrr@10",
                    "evidence_recall@5",
                    "all_evidence@5",
                }
            }
            | {"latency_p50_ms": values["latency"]["p50_ms"]}
            for arm, values in retrieval["arms"].items()
        }
    contexts = report.get("contexts", {}).get("contexts")
    if contexts:
        summary["contexts"] = {
            "samples": contexts["samples"],
            "top_k": contexts["top_k"],
            "retrieval_latency": contexts["retrieval_latency"],
        }
    answers = report.get("answers", {}).get("answers")
    if answers:
        summary["answers"] = {
            key: {
                metric: value["metrics"][metric]
                for metric in ("exact", "tolerant", "format_failure_rate")
                if metric in value["metrics"]
            }
            | {
                "coverage": value["coverage"],
                "latency_p50_ms": value["latency_generation"]["p50_ms"],
            }
            for key, value in answers["summary"].items()
        }
        summary["llm_calls"] = answers["llm_calls"]
    return summary


@app.function(
    gpu=GPU,
    cpu=4,
    memory=16384,
    volumes={"/data": data_volume, "/cache": cache_volume},
    secrets=[database_secret, llm_secret],
    timeout=3600,
    max_containers=1,
    single_use_containers=True,
    retries=0,
)
async def evaluate(commands: list[str], options: dict[str, Any]) -> dict[str, Any]:
    import secrets
    import tempfile
    import time
    from datetime import UTC, datetime
    from urllib.parse import urlsplit
    from uuid import uuid4

    from ledger_rag.evaluation.benchmarks.finqa import (
        bench_settings,
        build_parser,
        execute,
        validate_args,
    )
    from ledger_rag.retrieval.qdrant import QdrantRetriever

    started = time.perf_counter()
    if not commands or any(command not in GPU_COMMANDS for command in commands):
        raise ValueError("commands must be a non-empty subset of retrieval, contexts")
    if urlsplit(os.environ["LEDGER_QDRANT_URL"]).scheme != "https":
        raise ValueError("Evaluation requires an HTTPS Qdrant URL")
    run_id = uuid4().hex
    key = secrets.token_urlsafe(32)
    parser = build_parser()
    report: dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "git_sha": options.get("git_sha", "unknown"),
        "gpu": _gpu_name(),
        "gpu_preference": GPU,
        "tei_image": TEI_IMAGE,
        "read_only": True,
        "options": {k: v for k, v in options.items() if "key" not in k},
    }
    with tempfile.TemporaryDirectory(prefix="ledger-eval-") as directory:
        fetch = await asyncio.create_subprocess_exec(
            "bash", "/opt/ledger/fetch_finqa.sh", directory
        )
        if await fetch.wait() != 0:
            raise RuntimeError("Pinned FinQA fetch or checksum verification failed")
        parsed = {}
        for command in commands:
            args = parser.parse_args(benchmark_argv(command, options, directory, key))
            validate_args(parser, args)
            parsed[command] = args
        async with AsyncExitStack() as stack:
            tei_started = time.perf_counter()
            await stack.enter_async_context(
                tei_server(EMBEDDING_MODEL, EMBEDDING_REVISION, EMBED_PORT, key)
            )
            await stack.enter_async_context(
                tei_server(RERANKER_MODEL, RERANKER_REVISION, RERANK_PORT, key)
            )
            report["tei_startup_seconds"] = round(time.perf_counter() - tei_started, 1)
            store = QdrantRetriever(bench_settings(next(iter(parsed.values()))))
            try:
                count = await store.count()
            finally:
                await store.close()
            if count != EXPECTED_CHUNKS:
                raise RuntimeError(f"{COLLECTION} holds {count} points, not {EXPECTED_CHUNKS}")
            report["collection_points"] = count
            for command, args in parsed.items():
                print(f"evaluation.{command}.started run_id={run_id}", flush=True)
                command_started = time.perf_counter()
                report[command] = await execute(args)
                report[command]["wall_seconds"] = round(time.perf_counter() - command_started, 1)
    report["gpu_seconds"] = round(time.perf_counter() - started, 1)
    output = DATA / "evaluation" / f"{run_id}.json"
    report["report_path"] = str(output)
    await data_volume.reload.aio()  # type: ignore[no-untyped-call]
    await asyncio.to_thread(_write_report, output, report)
    await data_volume.commit.aio()  # type: ignore[no-untyped-call]
    return summarize_report(report)


@app.function(
    cpu=2,
    memory=8192,
    volumes={"/data": data_volume},
    secrets=[llm_secret],
    timeout=4 * 3600,
    max_containers=1,
    single_use_containers=True,
    retries=0,
)
async def generate(options: dict[str, Any], contexts_path: str) -> dict[str, Any]:
    """CPU-only LLM stage: answers over contexts saved by the GPU stage (no Qdrant access)."""
    import tempfile
    import time
    from datetime import UTC, datetime
    from uuid import uuid4

    from ledger_rag.evaluation.benchmarks.finqa import build_parser, execute, validate_args

    started = time.perf_counter()
    if not re.fullmatch(r"/data/evaluation/[0-9a-f]{32}\.json", contexts_path):
        raise ValueError("contexts_path must be an evaluation report on the data volume")
    source = Path(contexts_path)
    await data_volume.reload.aio()  # type: ignore[no-untyped-call]
    run_id = uuid4().hex
    report: dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "git_sha": options.get("git_sha", "unknown"),
        "gpu": None,
        "options": {k: v for k, v in options.items() if "key" not in k},
    }
    with tempfile.TemporaryDirectory(prefix="ledger-eval-") as directory:
        fetch = await asyncio.create_subprocess_exec(
            "bash", "/opt/ledger/fetch_finqa.sh", directory
        )
        if await fetch.wait() != 0:
            raise RuntimeError("Pinned FinQA fetch or checksum verification failed")
        parser = build_parser()
        args = parser.parse_args(
            benchmark_argv("answers", options, directory, "unused", str(source))
        )
        validate_args(parser, args)
        report["answers"] = await execute(args)
    report["cpu_seconds"] = round(time.perf_counter() - started, 1)
    output = DATA / "evaluation" / f"{run_id}.json"
    report["report_path"] = str(output)
    await asyncio.to_thread(_write_report, output, report)
    await data_volume.commit.aio()  # type: ignore[no-untyped-call]
    return summarize_report(report)


def _git_sha() -> str:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        sha += "-dirty"
    return sha


def _split(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


@app.local_entrypoint()
def main(
    command: str = "all",
    split: str = "dev",
    unseal_test: bool = False,
    limit: int = 0,
    configs: str = "sparse,dense,hybrid,hybrid:rerank",
    scopes: str = "company,global",
    models: str = "gemini-3.5-flash:low,gemini-3.5-flash-lite",
    answer_samples: int = 200,
    answer_modes: str = "retrieved@3,retrieved@5,oracle",
    temperature: str = "default",
    max_tokens: int = 2048,
    llm_concurrency: int = 4,
    llm_rpm: float = 0.0,
    retrieval_concurrency: int = 4,
    seed: int = 13,
    contexts_run: str = "",
) -> None:
    """``--command retrieval|contexts|answers|all``.

    answers = GPU contexts stage then CPU LLM stage; contexts = the GPU stage only.
    ``--contexts-run <run_id>`` reuses saved contexts and skips the GPU stage.
    """
    if command not in {"retrieval", "contexts", "answers", "all"}:
        raise SystemExit("--command must be retrieval, contexts, answers or all")
    if split == "test" and not unseal_test:
        raise SystemExit("the test split is sealed; pass --unseal-test after configuration freeze")
    options: dict[str, Any] = {
        "split": split,
        "unseal_test": unseal_test,
        "limit": limit,
        "configs": _split(configs),
        "scopes": _split(scopes),
        "models": _split(models),
        "answer_samples": answer_samples,
        "answer_modes": _split(answer_modes),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "llm_concurrency": llm_concurrency,
        "llm_rpm": llm_rpm or None,
        "retrieval_concurrency": retrieval_concurrency,
        "seed": seed,
        "git_sha": _git_sha(),
    }
    gpu_commands = []
    if command in {"retrieval", "all"}:
        gpu_commands.append("retrieval")
    if command == "contexts" or (command in {"answers", "all"} and not contexts_run):
        gpu_commands.append("contexts")
    contexts_path = f"/data/evaluation/{contexts_run}.json" if contexts_run else ""
    if gpu_commands:
        summary = evaluate.remote(gpu_commands, options)
        print(json.dumps(summary, indent=2))
        contexts_path = contexts_path or str(summary["report_path"])
    if command in {"answers", "all"}:
        print(json.dumps(generate.remote(options, contexts_path), indent=2))
