"""Serving measurements for the deployed ``ledger-llm-serve`` endpoint (``compute/llm.py``).

Runs inside Modal on a CPU container, so the bearer key comes from the
``ledger-llm-serve`` secret and never reaches the workstation. Reports are written
to ``ledger-data:/llm-serve/bench-<run_id>.json``.

    uv run --group compute modal run compute/llm_bench.py               # all phases
    uv run --group compute modal run compute/llm_bench.py --phases metrics

Phases:

- ``cold``: first request against a scaled-to-zero endpoint. The client-observed
  wait is split with the server's ``/ledger/startup`` epochs into scheduling plus
  image pull, container init, weights load, torch.compile, CUDA graphs and the rest.
- ``controls``: how thinking is actually controlled. ``max_tokens=1`` requests expose
  the rendered prompt length per variant (the template injects effort instructions),
  short streams show whether reasoning is emitted, and one question is answered at
  each effort to compare reasoning lengths.
- ``single``: warm single-stream time to first token and decode tokens/s.
- ``throughput``: aggregate output tokens/s at concurrency 1, 8 and 32.
- ``long``: one stream longer than Modal's 150 s HTTP window, to check that streamed
  responses are not cut.
- ``metrics``: cumulative speculative-decoding counters (MTP acceptance).

Every request uses temperature 0 and a unique nonce prefix so prefix caching cannot
inflate results; speculative-decoding acceptance is the delta of vLLM's Prometheus
counters around each phase.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import modal

SERVE_APP, SERVE_FUNCTION = "ledger-llm-serve", "serve"
KEY_ENV = "LEDGER_LLM_SERVE_API_KEY"
PHASES = ("cold", "controls", "single", "throughput", "long", "metrics")
NO_THINK: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}
CONCURRENCY_LEVELS = (1, 8, 32)
SPEC_COUNTERS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}
PER_POSITION = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
CONTROL_VARIANTS: dict[str, dict[str, Any]] = {
    "default": {},
    "enable_thinking_false": NO_THINK,
    "template_effort_low": {"chat_template_kwargs": {"reasoning_effort": "low"}},
    "template_effort_medium": {"chat_template_kwargs": {"reasoning_effort": "medium"}},
    "template_effort_xhigh": {"chat_template_kwargs": {"reasoning_effort": "xhigh"}},
    "request_effort_low": {"reasoning_effort": "low"},
    "request_effort_high": {"reasoning_effort": "high"},
    "low_with_budget_256": {
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "thinking_token_budget": 256,
    },
}
EFFORT_QUESTION = (
    "A company's revenue was $1,250 million in 2019 and $1,410 million in 2020. "
    "Operating expenses were $980 million and $1,050 million. By how many percentage "
    "points did the operating margin change? Reply with the final number only."
)

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("httpx==0.28.1")
app = modal.App("ledger-llm-bench", image=image)
data_volume = modal.Volume.from_name("ledger-data", create_if_missing=True)
serve_secret = modal.Secret.from_name(SERVE_APP, required_keys=[KEY_ENV])


def validate_serve_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not (parsed.hostname or "").endswith(".modal.run")
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
        or parsed.port not in (None, 443)
    ):
        raise ValueError("Benchmark endpoint must be a credential-free Modal HTTPS origin")


def parse_counters(text: str) -> dict[str, Any]:
    """Speculative-decoding counters summed over label sets, plus per-position accepts."""
    totals = dict.fromkeys(SPEC_COUNTERS, 0.0)
    positions: dict[int, float] = {}
    sample = re.compile(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|\+Inf)\s*$")
    for line in text.splitlines():
        match = sample.match(line)
        if not match:
            continue
        name, labels, value = match.group(1), match.group(2) or "", float(match.group(3))
        for key, metric in SPEC_COUNTERS.items():
            if name == metric:
                totals[key] += value
        if name == PER_POSITION:
            position = re.search(r'position="(\d+)"', labels)
            if position:
                index = int(position.group(1))
                positions[index] = positions.get(index, 0.0) + value
    return totals | {"per_position": positions}


def acceptance(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """MTP acceptance between two counter snapshots."""
    delta = {key: after[key] - before[key] for key in SPEC_COUNTERS}
    drafts, draft_tokens = delta["drafts"], delta["draft_tokens"]
    accepted = delta["accepted_tokens"]
    per_position = {
        str(index): round(
            (after["per_position"][index] - before["per_position"].get(index, 0.0)) / drafts, 3
        )
        for index in sorted(after["per_position"])
        if drafts
    }
    return {
        **{key: int(value) for key, value in delta.items()},
        "acceptance_rate": round(accepted / draft_tokens, 3) if draft_tokens else None,
        "mean_acceptance_length": round(1 + accepted / drafts, 3) if drafts else None,
        "per_position_acceptance": per_position,
    }


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, int(-(-q * len(ordered) // 1)) - 1))
    return round(ordered[index], 3)


def context(seed: int, rows: int) -> str:
    """Deterministic synthetic financial table (about 20 tokens per row)."""
    rng = random.Random(seed)
    items = (
        "net revenue",
        "cost of sales",
        "research and development",
        "selling general and administrative",
        "operating income",
        "interest expense",
        "income tax expense",
        "net income",
    )
    lines = ["| line item (in millions) | 2021 | 2020 | 2019 |", "| --- | --- | --- | --- |"]
    for index in range(rows):
        name = f"{items[index % len(items)]} segment {index // len(items) + 1}"
        values = " | ".join(f"{rng.randint(1000, 999999) / 10:,.1f}" for _ in range(3))
        lines.append(f"| {name} | {values} |")
    return "\n".join(lines)


def messages(prompt: str) -> list[dict[str, str]]:
    nonce = uuid.uuid4().hex
    return [{"role": "user", "content": f"[request {nonce}]\n{prompt}"}]


def analysis_prompt(seed: int, rows: int) -> str:
    return (
        f"{context(seed, rows)}\n\nUsing the table, describe the year-over-year trend of "
        "every line item in detail, with computed growth rates."
    )


async def stream_chat(client: httpx.AsyncClient, url: str, body: dict[str, Any]) -> dict[str, Any]:
    """One streamed chat completion with client-side timings."""
    payload = body | {"stream": True, "stream_options": {"include_usage": True}}
    sent = time.perf_counter()
    first: float | None = None
    last: float | None = None
    usage: dict[str, Any] = {}
    content = reasoning = 0
    finish: str | None = None
    try:
        async with client.stream("POST", f"{url}/v1/chat/completions", json=payload) as response:
            if response.status_code != 200:
                await response.aread()
                return {"status": response.status_code, "error": response.text[:200]}
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or ""
                    thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if text or thought:
                        last = time.perf_counter()
                        first = first or last
                    content += len(text)
                    reasoning += len(thought)
                    finish = choice.get("finish_reason") or finish
    except (httpx.HTTPError, ValueError) as exc:
        return {
            "status": None,
            "error": type(exc).__name__,
            "elapsed_s": time.perf_counter() - sent,
        }
    end = time.perf_counter()
    completion = int(usage.get("completion_tokens") or 0)
    decode = (
        (completion - 1) / (last - first)
        if first is not None and last is not None and last > first and completion >= 16
        else None
    )
    return {
        "status": 200,
        "ttft_s": round(first - sent, 4) if first is not None else None,
        "total_s": round(end - sent, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion,
        "decode_tokens_per_s": round(decode, 1) if decode else None,
        "finish_reason": finish,
        "content_chars": content,
        "reasoning_chars": reasoning,
    }


async def counters(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
    response = await client.get(f"{url}/metrics")
    response.raise_for_status()
    return parse_counters(response.text)


async def cold(client: httpx.AsyncClient, url: str, model: str) -> dict[str, Any]:
    started = time.time()
    attempts = 0
    statuses: dict[str, int] = {}
    while True:
        attempts += 1
        try:
            response = await client.get(f"{url}/v1/models", timeout=170)
            key = str(response.status_code)
            if response.status_code == 200:
                break
            if response.status_code in {401, 403, 404}:
                raise RuntimeError(f"/v1/models returned {response.status_code}")
        except httpx.TransportError as exc:
            key = type(exc).__name__
        statuses[key] = statuses.get(key, 0) + 1
        if time.time() - started > 1800:
            raise TimeoutError("endpoint not ready after 30 minutes")
        await asyncio.sleep(2)
    ready = time.time()
    startup = (await client.get(f"{url}/ledger/startup")).json()
    first = await stream_chat(
        client,
        url,
        {"model": model, "messages": messages("Say OK."), "max_tokens": 8, "temperature": 0}
        | NO_THINK,
    )
    events = startup.get("events", {})
    container = startup.get("container_started", ready)
    process = startup.get("process_started", ready)
    return {
        "already_warm": container < started,
        "probe_attempts": attempts,
        "probe_statuses": statuses,
        "client_observed_seconds": round(ready - started, 1),
        "split_seconds": {
            "scheduling_and_image_pull": round(container - started, 1),
            "container_init": round(process - container, 1),
            "weights_load": startup.get("weights_load_seconds"),
            "model_load_total": startup.get("model_load_seconds"),
            "torch_compile": startup.get("compile_seconds"),
            "compile_cache_load": startup.get("compile_cache_load_seconds"),
            "cuda_graph_capture": startup.get("cuda_graph_seconds"),
            "engine_init_total": startup.get("init_engine_seconds"),
            "process_to_ready": startup.get("ready_seconds_after_process"),
            "ready_to_client": round(ready - events["app_ready"], 1)
            if "app_ready" in events
            else None,
        },
        "startup": startup,
        "first_request": first,
    }


async def _plain(client: httpx.AsyncClient, url: str, body: dict[str, Any]) -> dict[str, Any]:
    response = await client.post(f"{url}/v1/chat/completions", json=body)
    if response.status_code != 200:
        return {"status": response.status_code, "error": response.text[:200]}
    usage = response.json().get("usage") or {}
    return {"status": 200, "prompt_tokens": usage.get("prompt_tokens")}


async def controls(client: httpx.AsyncClient, url: str, model: str) -> dict[str, Any]:
    question = [{"role": "user", "content": EFFORT_QUESTION}]
    variants: dict[str, Any] = {}
    for name, extra in CONTROL_VARIANTS.items():
        body = {"model": model, "messages": question, "temperature": 0} | extra
        rendered = await _plain(client, url, body | {"max_tokens": 1})
        short = await stream_chat(client, url, body | {"max_tokens": 64})
        variants[name] = {
            "status": rendered["status"],
            "prompt_tokens": rendered.get("prompt_tokens"),
            "error": rendered.get("error"),
            "reasoning_chars_in_64_tokens": short.get("reasoning_chars"),
            "content_chars_in_64_tokens": short.get("content_chars"),
        }
    efforts: dict[str, Any] = {}
    for name in (
        "enable_thinking_false",
        "template_effort_low",
        "low_with_budget_256",
        "template_effort_medium",
        "default",
    ):
        body = {"model": model, "messages": question, "temperature": 0, "max_tokens": 8192}
        efforts[name] = await stream_chat(client, url, body | CONTROL_VARIANTS[name])
    return {"variants": variants, "effort_lengths": efforts}


async def single(client: httpx.AsyncClient, url: str, model: str) -> dict[str, Any]:
    def body() -> dict[str, Any]:
        prompt = analysis_prompt(7, 100)
        return {"model": model, "messages": messages(prompt), "max_tokens": 512, "temperature": 0}

    await stream_chat(client, url, body() | NO_THINK)
    before = await counters(client, url)
    runs = [await stream_chat(client, url, body() | NO_THINK) for _ in range(5)]
    after = await counters(client, url)
    ok = [run for run in runs if run["status"] == 200]
    return {
        "requests": len(runs),
        "ok": len(ok),
        "prompt_tokens": [run["prompt_tokens"] for run in ok],
        "completion_tokens": [run["completion_tokens"] for run in ok],
        "ttft_p50_s": percentile([run["ttft_s"] for run in ok if run["ttft_s"]], 0.5),
        "ttft_max_s": percentile([run["ttft_s"] for run in ok if run["ttft_s"]], 1.0),
        "decode_tokens_per_s_p50": percentile(
            [run["decode_tokens_per_s"] for run in ok if run["decode_tokens_per_s"]], 0.5
        ),
        "speculative": acceptance(before, after),
    }


async def _drain(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    queue: asyncio.Queue[int],
    results: list[dict[str, Any]],
) -> None:
    while not queue.empty():
        index = queue.get_nowait()
        body = {
            "model": model,
            "messages": messages(analysis_prompt(100 + index, 60)),
            "max_tokens": 256,
            "temperature": 0,
        }
        results.append(await stream_chat(client, url, body | NO_THINK))


async def throughput(client: httpx.AsyncClient, url: str, model: str) -> dict[str, Any]:
    levels: dict[str, Any] = {}
    for level in CONCURRENCY_LEVELS:
        total = max(6, 4 * level)
        queue: asyncio.Queue[int] = asyncio.Queue()
        for index in range(total):
            queue.put_nowait(index)
        results: list[dict[str, Any]] = []
        before = await counters(client, url)
        started = time.perf_counter()
        await asyncio.gather(*(_drain(client, url, model, queue, results) for _ in range(level)))
        wall = time.perf_counter() - started
        after = await counters(client, url)
        ok = [run for run in results if run["status"] == 200]
        output = sum(run["completion_tokens"] for run in ok)
        prompt = sum(run["prompt_tokens"] or 0 for run in ok)
        ttfts = [run["ttft_s"] for run in ok if run["ttft_s"] is not None]
        levels[str(level)] = {
            "requests": total,
            "ok": len(ok),
            "wall_s": round(wall, 2),
            "output_tokens_per_s": round(output / wall, 1),
            "prompt_tokens_per_s": round(prompt / wall, 1),
            "requests_per_s": round(len(ok) / wall, 3),
            "mean_prompt_tokens": round(prompt / len(ok), 1) if ok else None,
            "ttft_p50_s": percentile(ttfts, 0.5),
            "ttft_p95_s": percentile(ttfts, 0.95),
            "per_stream_decode_tokens_per_s_p50": percentile(
                [run["decode_tokens_per_s"] for run in ok if run["decode_tokens_per_s"]], 0.5
            ),
            "speculative": acceptance(before, after),
        }
    return levels


async def long_stream(
    client: httpx.AsyncClient, url: str, model: str, decode_tps: float | None
) -> dict[str, Any]:
    tokens = min(30000, int((decode_tps or 120) * 200))
    body = {
        "model": model,
        "messages": messages("Write a very long story about a lighthouse keeper."),
        "max_tokens": tokens,
        "temperature": 0,
        "ignore_eos": True,
    }
    result = await stream_chat(client, url, body | NO_THINK)
    elapsed = result.get("total_s") or result.get("elapsed_s") or 0
    return {
        "requested_tokens": tokens,
        "result": result,
        "exceeded_150s": elapsed > 150,
        "completed": result.get("completion_tokens") == tokens,
    }


async def run(url: str, key: str, phases: Sequence[str]) -> dict[str, Any]:
    unknown = sorted(set(phases) - set(PHASES))
    if unknown:
        raise ValueError(f"unknown phases: {', '.join(unknown)}")
    url = url.rstrip("/")
    result: dict[str, Any] = {"run_id": uuid.uuid4().hex, "endpoint": url, "phases": list(phases)}
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {key}"},
        timeout=httpx.Timeout(1800, connect=30),
        follow_redirects=False,
        limits=httpx.Limits(max_connections=64),
    ) as client:
        if "cold" in phases:
            # Uses the alias-free name; the alias is verified below once the server is up.
            result["cold"] = await cold(client, url, "Qwen/Qwen3.8-27B-FP8")
        models = (await client.get(f"{url}/v1/models", timeout=1800)).json()
        served = sorted(item["id"] for item in models.get("data", []))
        model = next(name for name in served if "@" not in name)
        result["served_models"] = served
        result["server_version"] = (await client.get(f"{url}/version")).json().get("version")
        if "controls" in phases:
            result["controls"] = await controls(client, url, model)
        if "single" in phases:
            result["single"] = await single(client, url, model)
        if "throughput" in phases:
            result["throughput"] = await throughput(client, url, model)
        if "long" in phases:
            decode = (result.get("single") or {}).get("decode_tokens_per_s_p50")
            result["long"] = await long_stream(client, url, model, decode)
        if "metrics" in phases:
            snapshot = await counters(client, url)
            empty: dict[str, Any] = dict.fromkeys(SPEC_COUNTERS, 0.0) | {"per_position": {}}
            result["metrics"] = {"cumulative": acceptance(empty, snapshot)}
    result["finished_at"] = time.time()
    return result


@app.function(
    secrets=[serve_secret],
    volumes={"/data": data_volume},
    cpu=2,
    memory=8192,
    timeout=3600,
    max_containers=1,
    single_use_containers=True,
    retries=0,
)
async def measure(phases: list[str], url: str = "", deadline_unix: float = 0) -> dict[str, Any]:
    remaining = deadline_unix - time.time() if deadline_unix else 3600
    if not math.isfinite(remaining) or not 0 < remaining <= 3600:
        raise ValueError("Benchmark session deadline is invalid or expired")
    resolved = url or await modal.Function.from_name(SERVE_APP, SERVE_FUNCTION).get_web_url.aio()
    if not resolved:
        raise RuntimeError(f"{SERVE_APP} is not deployed")
    validate_serve_url(resolved)
    result = await asyncio.wait_for(run(resolved, os.environ[KEY_ENV], phases), timeout=remaining)
    output = Path("/data/llm-serve") / f"bench-{result['run_id']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    await data_volume.commit.aio()  # type: ignore[no-untyped-call]
    result["report_path"] = str(output)
    return result


@app.local_entrypoint()
def main(phases: str = "cold,controls,single,throughput,long,metrics") -> None:
    print(json.dumps(measure.remote([phase for phase in phases.split(",") if phase]), indent=2))
