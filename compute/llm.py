"""Self-hosted open-weight LLM on Modal: vLLM's OpenAI-compatible server.

Serves ``Qwen/Qwen3.8-27B-FP8`` (Apache-2.0, dense 27.8B, ``Qwen3_5ForConditionalGeneration``,
~31 GB FP8 checkpoint) on one H100, pinned to an immutable Hub commit, following the
vLLM recipe (https://recipes.vllm.ai/Qwen/Qwen3.8-27B): ``--reasoning-parser qwen3``
(the chat template opens the assistant turn with ``<think>``), ``qwen3_coder`` tool
calls, FP8 KV cache, ``--language-model-only`` (no vision encoder, larger KV pool) and
MTP speculative decoding (``LEDGER_LLM_SPECULATIVE_TOKENS``; 0 disables it).

Thinking is on by default. Disable it per request with
``extra_body={"chat_template_kwargs": {"enable_thinking": false}}``; the template also
accepts ``chat_template_kwargs.reasoning_effort`` of ``low``, ``medium`` or ``xhigh``.

Operations::

    scripts/sync_modal_secrets.sh llm-serve          # Vault -> Modal secret ledger-llm-serve
    uv run --group compute modal run compute/llm.py::download_weights
    uv run --group compute modal deploy compute/llm.py
    uv run --group compute modal run compute/llm_bench.py     # serving measurements

Endpoint: ``https://<workspace>--ledger-llm-serve-serve.modal.run/v1`` with
``Authorization: Bearer <LEDGER_LLM_SERVE_API_KEY>``. ``GET /health`` is the only
unauthenticated route; ``/metrics`` (Prometheus) and ``/ledger/startup`` (cold-start
breakdown) need the key. See ``compute/llm_middleware.py``.

Cost guardrails: ``min_containers=0`` and a 5-minute ``scaledown_window`` (scale to
zero), ``max_containers=1``. Requests wait through a cold start (several minutes).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import modal

CONTAINER_STARTED = time.time()
HERE = Path(__file__).resolve().parent

APP_NAME = "ledger-llm-serve"
MODEL = "Qwen/Qwen3.8-27B-FP8"
MODEL_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
# The revision alias lets clients verify the served weights through GET /v1/models.
SERVED_MODEL_NAMES = (MODEL, f"{MODEL}@{MODEL_REVISION}")
VLLM_VERSION = "0.32.0"
# vLLM 0.32 wheels use torch 2.13 / CUDA 13; Modal's driver supports CUDA 13.0.
CUDA_IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04"
# "H100!" prevents an automatic upgrade to H200 so serving measurements stay comparable.
GPU = "H100!"
PORT = 8000
MAX_MODEL_LEN = 32768
GPU_MEMORY_UTILIZATION = 0.90
MAX_NUM_SEQS = 128
SCALEDOWN_SECONDS = 300
STARTUP_TIMEOUT_SECONDS = 30 * 60
REQUEST_TIMEOUT_SECONDS = 30 * 60
SPECULATIVE_ENV = "LEDGER_LLM_SPECULATIVE_TOKENS"
SPECULATIVE_TOKENS = int(os.environ.get(SPECULATIVE_ENV, "3"))
KEY_ENV = "LEDGER_LLM_SERVE_API_KEY"
HF_CACHE = "/root/.cache/huggingface"
VLLM_CACHE = "/root/.cache/vllm"
STARTUP_REPORT = "/run/ledger/startup.json"


def vllm_argv(speculative_tokens: int = SPECULATIVE_TOKENS) -> list[str]:
    """The ``vllm serve`` command line (pure, so tests can pin every flag)."""
    if not 0 <= speculative_tokens <= 5:
        raise ValueError("speculative tokens must be within [0, 5]")
    argv = [
        "vllm",
        "serve",
        MODEL,
        "--revision",
        MODEL_REVISION,
        "--tokenizer-revision",
        MODEL_REVISION,
        "--served-model-name",
        *SERVED_MODEL_NAMES,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--kv-cache-dtype",
        "fp8",
        "--language-model-only",
        "--reasoning-parser",
        "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--middleware",
        "llm_middleware.BearerAuth",
        "--disable-uvicorn-access-log",
        "--disable-fastapi-docs",
    ]
    if speculative_tokens:
        config = {"method": "mtp", "num_speculative_tokens": speculative_tokens}
        argv += ["--speculative-config", json.dumps(config, separators=(",", ":"))]
    return argv


serve_image = (
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
            "PYTHONPATH": "/root",
        }
    )
    .add_local_file(HERE / "llm_middleware.py", "/root/llm_middleware.py")
)
app = modal.App(APP_NAME)
hf_volume = modal.Volume.from_name("ledger-llm-hf-cache", create_if_missing=True)
vllm_volume = modal.Volume.from_name("ledger-llm-vllm-cache", create_if_missing=True)
serve_secret = modal.Secret.from_name(APP_NAME, required_keys=[KEY_ENV])


@app.function(image=serve_image, volumes={HF_CACHE: hf_volume}, cpu=4, memory=8192, timeout=3600)
def download_weights() -> dict[str, Any]:
    """Fetch the pinned snapshot into the cache volume (CPU only; run before deploy)."""
    from huggingface_hub import snapshot_download

    started = time.perf_counter()
    path = Path(snapshot_download(MODEL, revision=MODEL_REVISION))
    files = [item for item in path.rglob("*") if item.is_file()]
    hf_volume.commit()
    return {
        "model": MODEL,
        "revision": MODEL_REVISION,
        "snapshot": str(path),
        "files": len(files),
        "gigabytes": round(sum(item.stat().st_size for item in files) / 1e9, 2),
        "seconds": round(time.perf_counter() - started, 1),
    }


def _gpu_name() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@app.function(
    image=serve_image,
    gpu=GPU,
    cpu=8,
    memory=64 * 1024,
    volumes={HF_CACHE: hf_volume, VLLM_CACHE: vllm_volume},
    secrets=[serve_secret],
    env={SPECULATIVE_ENV: str(SPECULATIVE_TOKENS), "HF_HUB_OFFLINE": "1"},
    min_containers=0,
    max_containers=1,
    scaledown_window=SCALEDOWN_SECONDS,
    timeout=REQUEST_TIMEOUT_SECONDS,
)
@modal.concurrent(max_inputs=256, target_inputs=128)
@modal.web_server(port=PORT, startup_timeout=STARTUP_TIMEOUT_SECONDS)
def serve() -> None:
    """Start vLLM; its stdout is re-emitted as JSON lines and mined for startup timings."""
    import threading

    from llm_middleware import (
        KEY_DIGEST_ENV,
        STARTUP_REPORT_ENV,
        StartupTimings,
        key_digest,
        log_event,
        structured_line,
    )

    # Only a digest reaches vLLM: the raw key never enters its environment or logs.
    environment = {name: value for name, value in os.environ.items() if name != KEY_ENV}
    environment[KEY_DIGEST_ENV] = key_digest(os.environ[KEY_ENV])
    environment[STARTUP_REPORT_ENV] = STARTUP_REPORT
    argv = vllm_argv(int(os.environ.get(SPECULATIVE_ENV, str(SPECULATIVE_TOKENS))))
    report = Path(STARTUP_REPORT)
    timings = StartupTimings(container_started=CONTAINER_STARTED, process_started=time.time())
    timings.report.update(
        gpu=_gpu_name(),
        model=MODEL,
        revision=MODEL_REVISION,
        vllm_version=VLLM_VERSION,
        argv=argv[2:],
    )
    timings.write(report)
    log_event("vllm_starting", model=MODEL, revision=MODEL_REVISION, gpu=timings.report["gpu"])
    process = subprocess.Popen(
        argv,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def pump() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            if not line.strip():
                continue  # progress bars emit bare carriage returns
            print(structured_line(line), flush=True)
            if timings.observe(line):
                timings.write(report)
        returncode = process.wait()
        log_event("vllm_exited", returncode=returncode)
        # Fail fast: without this a crashed server would hold the GPU until startup_timeout.
        os._exit(returncode or 1)

    threading.Thread(target=pump, name="vllm-log-pump", daemon=True).start()
