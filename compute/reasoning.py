"""Colocated reasoning benchmark: vLLM on loopback plus the benchmark, one GPU container.

The container downloads pinned weights into a cached Volume, starts ``vllm serve`` on
127.0.0.1 (no network exposure, so no key), fetches the pinned FinQA splits (SHA-256
verified), runs :mod:`ledger_rag.evaluation.benchmarks.finqa_reasoning` against it,
writes the report to ``ledger-data:/reasoning/<run_id>.json`` and exits. Nothing is
deployed and the GPU is released when the function returns; ``timeout`` bounds the
worst case. Cost is the container's GPU time, reported per run.

    ledgerctl modal -- run compute/reasoning.py --model qwen3.5-4b --limit 200 --samples 5

Models are pinned by Hub commit in :data:`MODELS`; add new ones there, never by name only.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, NamedTuple

import modal

HERE = Path(__file__).resolve().parent
VLLM_VERSION = "0.32.0"
CUDA_IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04"
HF_CACHE = "/root/.cache/huggingface"
VLLM_CACHE = "/root/.cache/vllm"
DATA = Path("/data")
PORT = 8000
STARTUP_SECONDS = 20 * 60
# Hard ceiling per container; with H100 at ~$3.95/h this caps a run near $3.30.
TIMEOUT_SECONDS = 50 * 60


class Served(NamedTuple):
    repo: str
    revision: str
    gpu: str
    max_model_len: int = 16384
    extra: tuple[str, ...] = ()


MODELS: dict[str, Served] = {
    # Small dense readers (Apache-2.0): the "less capable model" arms.
    "qwen3.5-4b": Served("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "L40S"),
    "qwen3.5-9b": Served("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "L40S"),
    # The reader of the 2026-11-01 baseline (docs/EVALUATION.md).
    "qwen3.8-27b-fp8": Served(
        "Qwen/Qwen3.8-27B-FP8",
        "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
        "H100!",
        extra=("--kv-cache-dtype", "fp8"),
    ),
}


def vllm_argv(served: Served) -> list[str]:
    """Loopback-only server; text model only, Qwen reasoning parser, prefix caching on."""
    return [
        "vllm",
        "serve",
        served.repo,
        "--revision",
        served.revision,
        "--tokenizer-revision",
        served.revision,
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--max-model-len",
        str(served.max_model_len),
        "--gpu-memory-utilization",
        "0.90",
        "--max-num-seqs",
        "256",
        "--enable-prefix-caching",
        "--language-model-only",
        "--reasoning-parser",
        "qwen3",
        "--disable-uvicorn-access-log",
        *served.extra,
    ]


def benchmark_argv(options: dict[str, Any], served: Served, data_dir: Path) -> list[str]:
    """Arguments for ``finqa_reasoning``; the test split is never reachable from here."""
    argv = [
        "--data-dir",
        str(data_dir),
        "--split",
        "dev",
        "--reader",
        "endpoint",
        "--llm-base-url",
        f"http://127.0.0.1:{PORT}/v1",
        "--llm-model",
        served.repo,
        "--llm-endpoint-kind",
        "self-hosted",
        "--llm-concurrency",
        str(int(options.get("concurrency", 128))),
        "--llm-max-tokens",
        str(int(options.get("max_tokens", 1024))),
        "--samples",
        str(int(options.get("samples", 1))),
        "--exemplars",
        str(int(options.get("exemplars", 4))),
        "--sample-temperature",
        str(float(options.get("sample_temperature", 0.7))),
        "--seed",
        str(int(options.get("seed", 13))),
    ]
    if options.get("limit"):
        argv += ["--limit", str(int(options["limit"]))]
    if options.get("debug_replies"):
        argv += ["--debug-replies", str(int(options["debug_replies"]))]
    if options.get("benchmark"):
        argv += ["--benchmark", str(options["benchmark"])]
    thinking = options.get("thinking", "off")
    kwargs = (
        {"enable_thinking": False} if thinking == "off" else {"reasoning_effort": str(thinking)}
    )
    argv += ["--llm-extra-body", json.dumps({"chat_template_kwargs": kwargs})]
    return argv


image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}")
    .env(
        {
            "HF_HOME": HF_CACHE,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_CACHE_ROOT": VLLM_CACHE,
            "TRITON_CACHE_DIR": f"{VLLM_CACHE}/triton",
            "FLASHINFER_WORKSPACE_BASE": f"{VLLM_CACHE}/flashinfer",
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
            "NO_PROXY": "127.0.0.1,localhost",
        }
    )
    .add_local_python_source("ledger_rag")
)
app = modal.App("ledger-reasoning", image=image)
hf_volume = modal.Volume.from_name("ledger-llm-hf-cache", create_if_missing=True)
vllm_volume = modal.Volume.from_name("ledger-llm-vllm-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("ledger-data", create_if_missing=True)


def _wait_ready(process: subprocess.Popen[bytes], deadline: float) -> None:
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as reply:
                if reply.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(3)
    raise TimeoutError("vLLM did not become ready")


def _gpu_name() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@app.function(
    gpu="L40S",
    volumes={HF_CACHE: hf_volume, VLLM_CACHE: vllm_volume, str(DATA): data_volume},
    timeout=TIMEOUT_SECONDS,
    cpu=4,
    memory=32768,
)
def run(model: str, options: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    from ledger_rag.evaluation.benchmarks import finqa_reasoning
    from ledger_rag.evaluation.datasets.fetch import fetch_verified

    served = MODELS[model]
    started = time.time()
    timings: dict[str, float] = {}
    snapshot_download(served.repo, revision=served.revision)
    hf_volume.commit()
    timings["weights_s"] = round(time.time() - started, 1)
    benchmark = str(options.get("benchmark") or "finqa")
    data_dir = DATA / "datasets" / benchmark
    data_dir.mkdir(parents=True, exist_ok=True)
    fetch_verified(benchmark, data_dir)
    process = subprocess.Popen(vllm_argv(served), env=os.environ.copy())
    try:
        _wait_ready(process, time.time() + STARTUP_SECONDS)
        vllm_volume.commit()
        timings["ready_s"] = round(time.time() - started, 1)
        parser = finqa_reasoning.build_parser()
        args = parser.parse_args(benchmark_argv(options, served, data_dir))
        finqa_reasoning.validate(parser, args)
        import asyncio

        report = asyncio.run(finqa_reasoning.run(args))
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
    timings["total_s"] = round(time.time() - started, 1)
    run_id = uuid.uuid4().hex
    report["run_id"] = run_id
    report["compute"] = {
        "model_key": model,
        "repo": served.repo,
        "revision": served.revision,
        "vllm": VLLM_VERSION,
        "gpu": _gpu_name(),
        "timings": timings,
        "options": options,
    }
    debug = report.pop("debug_replies", None)
    if debug is not None:
        # Raw replies over public benchmark pages, kept apart from the report.
        debug_path = DATA / "reasoning" / "debug" / f"{run_id}.json"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(json.dumps(debug, indent=2) + "\n")
    out = DATA / "reasoning" / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    data_volume.commit()
    return {
        "run_id": run_id,
        "brief": finqa_reasoning.brief(report),
        "summary": report["summary"],
        "compute": report["compute"],
        "elapsed_seconds": report["elapsed_seconds"],
        "provider_stats": report["provider_stats"],
    }


@app.local_entrypoint()
def main(
    model: str = "qwen3.5-4b",
    limit: int = 0,
    samples: int = 1,
    exemplars: int = 4,
    thinking: str = "off",
    max_tokens: int = 1024,
    sample_temperature: float = 0.7,
    concurrency: int = 128,
    benchmark: str = "finqa",
    debug_replies: int = 0,
    output: str = "",
) -> None:
    if model not in MODELS:
        raise SystemExit(f"unknown model {model!r}; choose from {', '.join(sorted(MODELS))}")
    options = {
        "limit": limit or None,
        "samples": samples,
        "exemplars": exemplars,
        "thinking": thinking,
        "max_tokens": max_tokens,
        "sample_temperature": sample_temperature,
        "concurrency": concurrency,
        "benchmark": benchmark,
        "debug_replies": debug_replies,
    }
    function = run.with_options(gpu=MODELS[model].gpu)
    result = function.remote(model, options)
    print(json.dumps(result, indent=2))
    if output:
        Path(output).write_text(json.dumps(result, indent=2) + "\n")
