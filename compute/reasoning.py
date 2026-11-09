"""Raw model vs Ledger backend, for any registered model, on Modal GPUs.

One container per model: it downloads the pinned weights into a cached Volume,
serves them with vLLM on 127.0.0.1 (no network exposure, so no key), fetches each
benchmark's pinned data (SHA-256 verified) and runs every requested arm of
:mod:`ledger_rag.evaluation.benchmarks.reasoning` against that one server: the raw
model and the model inside the Ledger pipeline see identical questions under
identical generation settings. Reports go to ``ledger-data:/reasoning/<run_id>.json``.
Several models run in parallel containers; nothing is deployed, and ``timeout``
bounds each container's cost.

    ledgerctl modal -- run compute/reasoning.py --models qwen3.5-4b,phi-4-mini \\
        --benchmarks finqa,tatqa --limit 200

Models come from ``config/models/<id>.toml`` (:mod:`ledger_rag.evaluation.models`);
hosted models (Gemini, ...) are run locally with the same benchmark CLI instead.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import modal

VLLM_VERSION = "0.32.0"
CUDA_IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04"
HF_CACHE = "/root/.cache/huggingface"
VLLM_CACHE = "/root/.cache/vllm"
DATA = Path("/data")
PORT = 8000
STARTUP_SECONDS = 20 * 60
# Hard ceiling per container: about $3.70 on an H100, $2.00 on an L40S at list rates.
TIMEOUT_SECONDS = 50 * 60
BENCHMARK_CHOICES = ("finqa", "convfinqa", "tatqa")


def vllm_argv(spec: dict[str, Any]) -> list[str]:
    """Loopback-only server for a registry spec, with prefix caching for shared prompts."""
    return [
        "vllm",
        "serve",
        spec["served_name"],
        "--revision",
        spec["revision"],
        "--tokenizer-revision",
        spec["revision"],
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--max-model-len",
        str(spec["max_model_len"]),
        "--gpu-memory-utilization",
        "0.90",
        "--max-num-seqs",
        "256",
        "--enable-prefix-caching",
        "--disable-uvicorn-access-log",
        *spec["vllm_args"],
    ]


def benchmark_argv(
    benchmark: str, options: dict[str, Any], spec_path: Path, data_dir: Path
) -> list[str]:
    """Arguments for the benchmark CLI; test needs ``unseal_test`` and a dev-frozen config."""
    split = str(options.get("split", "dev"))
    argv = [
        "--benchmark",
        benchmark,
        "--data-dir",
        str(data_dir),
        "--split",
        split,
        "--reader",
        "endpoint",
        "--model-spec",
        str(spec_path),
        "--llm-base-url",
        f"http://127.0.0.1:{PORT}/v1",
        "--llm-concurrency",
        str(int(options.get("concurrency", 128))),
        "--pipelines",
        ",".join(options.get("pipelines", ["raw", "ledger"])),
        "--samples",
        str(int(options.get("samples", 5))),
        "--exemplars",
        str(int(options.get("exemplars", 4))),
        "--sample-temperature",
        str(float(options.get("sample_temperature", 0.7))),
        "--seed",
        str(int(options.get("seed", 13))),
    ]
    if split == "test":
        if not options.get("unseal_test"):
            raise ValueError("the test split is sealed; pass --unseal-test with a frozen config")
        argv.append("--unseal-test")
    if options.get("mode"):
        argv += ["--mode", str(options["mode"])]
    if options.get("limit"):
        argv += ["--limit", str(int(options["limit"]))]
    if options.get("debug_replies"):
        argv += ["--debug-replies", str(int(options["debug_replies"]))]
    if options.get("contexts"):
        # Saved retrieval contexts on the ledger-data volume (FinQA retrieved setting).
        argv += ["--contexts", str(DATA / str(options["contexts"]))]
    return argv


image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}")
    # The official TAT-QA scorer (ledger_rag.evaluation.official); versions match uv.lock.
    .uv_pip_install("scipy==1.18.1", "pandas==3.1.0")
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


def _save(report: dict[str, Any], run_id: str) -> None:
    debug = report.pop("debug_replies", None)
    if debug is not None:
        # Raw replies over public benchmark data, kept apart from the report.
        debug_path = DATA / "reasoning" / "debug" / f"{run_id}.json"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(json.dumps(debug, indent=2) + "\n")
    out = DATA / "reasoning" / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")


@app.function(
    gpu="L40S",
    volumes={HF_CACHE: hf_volume, VLLM_CACHE: vllm_volume, str(DATA): data_volume},
    timeout=TIMEOUT_SECONDS,
    cpu=4,
    memory=32768,
)
def run(spec: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    from ledger_rag.evaluation.benchmarks import reasoning as benchmark
    from ledger_rag.evaluation.datasets.fetch import fetch_verified

    started = time.time()
    timings: dict[str, float] = {}
    snapshot_download(spec["served_name"], revision=spec["revision"])
    hf_volume.commit()
    timings["weights_s"] = round(time.time() - started, 1)
    spec_path = Path("/tmp/model-spec.json")
    spec_path.write_text(json.dumps(spec))
    process = subprocess.Popen(vllm_argv(spec), env=os.environ.copy())
    reports: list[dict[str, Any]] = []
    try:
        _wait_ready(process, time.time() + STARTUP_SECONDS)
        vllm_volume.commit()
        timings["ready_s"] = round(time.time() - started, 1)
        for name in options["benchmarks"]:
            data_dir = DATA / "datasets" / name
            data_dir.mkdir(parents=True, exist_ok=True)
            fetch_verified(name, data_dir)
            parser = benchmark.build_parser()
            args = parser.parse_args(benchmark_argv(name, options, spec_path, data_dir))
            benchmark.validate(parser, args)
            report = asyncio.run(benchmark.run(args))
            run_id = uuid.uuid4().hex
            report["run_id"] = run_id
            report["compute"] = {
                "vllm": VLLM_VERSION,
                "gpu": _gpu_name(),
                "options": options,
                "container_elapsed_s": round(time.time() - started, 1),
            }
            _save(report, run_id)
            data_volume.commit()
            reports.append(report)
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
    timings["total_s"] = round(time.time() - started, 1)
    return {
        "model": spec["id"],
        "gpu": _gpu_name(),
        "timings": timings,
        "runs": [
            {
                "run_id": r["run_id"],
                "benchmark": r["benchmark"],
                "brief": benchmark.brief(r),
                "arms": {k: v["metrics"] for k, v in r["arms"].items()},
                "paired": r["paired"],
                "outcomes": {k: v["sample_outcomes"] for k, v in r["arms"].items()},
                "elapsed_seconds": r["elapsed_seconds"],
            }
            for r in reports
        ],
    }


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


@app.local_entrypoint()
def main(
    models: str = "qwen3.5-4b",
    benchmarks: str = "finqa",
    pipelines: str = "raw,ledger",
    limit: int = 0,
    samples: int = 5,
    exemplars: int = 4,
    mode: str = "",
    sample_temperature: float = 0.7,
    concurrency: int = 128,
    debug_replies: int = 0,
    split: str = "dev",
    unseal_test: bool = False,
    contexts: str = "",
    output: str = "",
) -> None:
    from ledger_rag.evaluation.models import load

    specs = [load(model_id) for model_id in _csv(models)]
    hosted = [spec.id for spec in specs if spec.kind != "vllm"]
    if hosted:
        raise SystemExit(f"hosted models run locally, not on Modal: {', '.join(hosted)}")
    chosen = _csv(benchmarks)
    unknown = sorted(set(chosen) - set(BENCHMARK_CHOICES))
    if unknown:
        raise SystemExit(f"unknown benchmarks: {', '.join(unknown)}")
    options = {
        "benchmarks": chosen,
        "pipelines": _csv(pipelines),
        "limit": limit or None,
        "samples": samples,
        "exemplars": exemplars,
        "mode": mode or None,
        "sample_temperature": sample_temperature,
        "concurrency": concurrency,
        "debug_replies": debug_replies,
        "split": split,
        "unseal_test": unseal_test,
        "contexts": contexts or None,
    }
    if contexts and chosen != ["finqa"]:
        raise SystemExit("--contexts applies to --benchmarks finqa only")
    if split not in {"dev", "test"} or (split == "test" and not unseal_test):
        raise SystemExit("use --split dev, or --split test with --unseal-test (frozen config)")
    calls = [run.with_options(gpu=spec.gpu).spawn(spec.manifest(), options) for spec in specs]
    results = []
    for spec, call in zip(specs, calls, strict=True):
        try:
            results.append(call.get())
        except Exception as exc:  # One model failing must not discard the others.
            results.append(
                {"model": spec.id, "error": type(exc).__name__, "detail": str(exc)[:300]}
            )
    for result in results:
        for item in result.get("runs", []):
            print(item["brief"])
        if "error" in result:
            print(f"{result['model']}: FAILED {result['error']}")
    if output:
        Path(output).write_text(json.dumps(results, indent=2) + "\n")
