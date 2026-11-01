"""Admission and attached ephemeral orchestration; dry-run unless explicitly approved."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTEXTS_RUN = "b29513c36fbf477cbf83ea3981bc41fa"
MODEL = "Qwen/Qwen3.8-27B-FP8"
REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
MAX_SECONDS = 9000
TOTAL_USD = 28.0
# Official list rates, fetched 2026-11-01; 2x reserve for actual-use billing/overhead.
RATE_PER_SECOND = 2 * (0.001097 + 10 * 0.0000131 + 72 * 0.00000222)
ARMS = [
    {
        "label": "qwen:off",
        "model": MODEL,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    {
        "label": "qwen:low",
        "model": MODEL,
        "extra_body": {"chat_template_kwargs": {"reasoning_effort": "low"}},
    },
]


def admission(seconds: int, spent: float) -> dict[str, Any]:
    if not 1 <= seconds <= MAX_SECONDS:
        raise ValueError("elapsed cap must be in [1, 9000] seconds")
    if not math.isfinite(spent) or spent < 0:
        raise ValueError("prior session spending must be finite and nonnegative")
    reserve = seconds * RATE_PER_SECOND
    if spent + reserve > TOTAL_USD:
        raise ValueError("conservative admission exceeds the TOTAL $28 session budget")
    return {
        "elapsed_cap_seconds": seconds,
        "prior_session_spend_usd": spent,
        "reserved_usd": round(reserve, 4),
        "remaining_contingency_usd": round(TOTAL_USD - spent - reserve, 4),
        "contexts_run": CONTEXTS_RUN,
        "calls": 1200,
        "gpu_containers": 1,
        "cpu_containers": 1,
        "test_unsealed": False,
    }


def reserve_budget(previous: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    if previous and (
        set(previous) != {"reserved_seconds", "reserved_usd", "prior_session_spend_usd"}
        or any(
            not isinstance(value, int | float) or not math.isfinite(value) or value < 0
            for value in previous.values()
        )
    ):
        raise ValueError("persistent session ledger is invalid; do not reset it")
    seconds = previous.get("reserved_seconds", 0) + plan["elapsed_cap_seconds"]
    reserved = previous.get("reserved_usd", 0) + plan["reserved_usd"]
    spent = max(previous.get("prior_session_spend_usd", 0), plan["prior_session_spend_usd"])
    if seconds > MAX_SECONDS or reserved + spent > TOTAL_USD:
        raise ValueError("persistent session ledger rejects cumulative admission")
    return {
        "reserved_seconds": seconds,
        "reserved_usd": reserved,
        "prior_session_spend_usd": spent,
    }


def selected_pairs(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    pairs = json.loads(path.read_text())
    valid = {
        (arm["label"], mode) for arm in ARMS for mode in ("retrieved@3", "retrieved@5", "oracle")
    }
    if (
        not isinstance(pairs, list)
        or len(pairs) != 2
        or any(
            not isinstance(pair, dict)
            or set(pair) != {"label", "mode"}
            or not isinstance(pair["label"], str)
            or not isinstance(pair["mode"], str)
            or (pair["label"], pair["mode"]) not in valid
            for pair in pairs
        )
        or pairs[0] == pairs[1]
    ):
        raise ValueError(
            "selection must contain exactly two distinct approved dev label/mode pairs"
        )
    return pairs


def bounded_call(function: Any, deadline: float, *args: Any) -> Any:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("session elapsed cap reached before submission")
    call = function.with_options(
        cpu=(2, 2),
        memory=(8192, 8192),
        timeout=math.ceil(remaining),
        retries=0,
        max_containers=1,
    ).spawn(*args)
    try:
        return call.get(timeout=max(0.001, deadline - time.monotonic()))
    finally:
        # Cancellation terminates CPU containers and any requests still on that worker.
        call.cancel(terminate_containers=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elapsed-cap-seconds", type=int, default=4500)
    parser.add_argument("--prior-session-spend-usd", type=float, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selected-pairs", type=Path, help="Full-dev only, after pilot review")
    parser.add_argument("--execute-approved-plan", action="store_true")
    args = parser.parse_args()
    plan = admission(args.elapsed_cap_seconds, args.prior_session_spend_usd)
    pairs = selected_pairs(args.selected_pairs)
    if pairs:
        plan.update(calls=1766, selected_pairs=pairs)
    if not args.execute_approved_plan:
        print(json.dumps(plan, indent=2))
        return
    manifest = args.manifest.resolve()
    if not manifest.is_relative_to(ROOT) or manifest.exists():
        raise ValueError("manifest must be a new file within this worktree")
    ledger = ROOT / ".scratch/modal-budget-session.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a+") as lock:
        # Held through shutdown: a second supervisor cannot admit another H100.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.seek(0)
        contents = lock.read()
        previous = json.loads(contents) if contents else {}
        updated = reserve_budget(previous, plan)
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps(updated, indent=2))
        lock.flush()
        os.fsync(lock.fileno())
        execute_session(plan, manifest, pairs)


def execute_session(plan: dict[str, Any], manifest: Path, pairs: list[dict[str, str]]) -> None:
    started = time.monotonic()
    deadline = started + plan["elapsed_cap_seconds"]
    os.environ["LEDGER_LLM_SESSION_DEADLINE_UNIX"] = str(time.time() + plan["elapsed_cap_seconds"])
    sys.path.insert(0, str(ROOT))
    import modal

    from compute import evaluation, llm, llm_bench

    app = modal.App("ledger-bounded-dev-session")
    app.include(llm.server_app).include(evaluation.answer_app).include(llm_bench.app)
    plan.update(
        started_at_unix=time.time(),
        deadline_unix=float(os.environ["LEDGER_LLM_SESSION_DEADLINE_UNIX"]),
        status="admitted",
        git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(plan, indent=2))
    try:
        # Never detach or deploy. Modal stops the whole App on exit/client disconnect.
        with modal.enable_output(), app.run(detach=False):
            plan["app_id"] = app.app_id
            manifest.write_text(json.dumps(plan, indent=2))
            url = llm.serve.get_web_url()
            if not url:
                raise RuntimeError("ephemeral serving URL unavailable")
            if not pairs:
                plan["benchmark"] = bounded_call(
                    llm_bench.measure,
                    min(deadline, started + 1200),
                    ["single", "throughput", "long", "metrics"],
                    url,
                    min(plan["deadline_unix"], plan["started_at_unix"] + 1200),
                )
            options = {
                "split": "dev",
                "unseal_test": False,
                "answer_samples": 200,
                "answer_sample_strategy": "stratified",
                "answer_modes": ["retrieved@3", "retrieved@5", "oracle"],
                "seed": 13,
                "temperature": "0",
                "max_tokens": 2048,
                "llm_concurrency": 32,
                "llm_endpoint_kind": "self-hosted",
                "llm_base_url": url.rstrip("/") + "/v1",
                "llm_model_revision": REVISION,
                "arms": ARMS,
                "llm_request_timeout": 120,
                "llm_deadline": 180,
                "llm_ready_timeout": 1200,
                "git_sha": plan["git_sha"],
                "session_deadline_unix": plan["deadline_unix"],
            }
            contexts = f"/data/evaluation/{CONTEXTS_RUN}.json"
            if pairs:
                plan["full_dev"] = []
                for pair in pairs:
                    arm = next(arm for arm in ARMS if arm["label"] == pair["label"])
                    options.update(answer_samples=883, arms=[arm], answer_modes=[pair["mode"]])
                    plan["full_dev"].append(
                        bounded_call(evaluation.generate, deadline, options, contexts)
                    )
                    manifest.write_text(json.dumps(plan, indent=2))
            else:
                plan["pilot"] = bounded_call(evaluation.generate, deadline, options, contexts)
            plan["status"] = "completed"
    except BaseException:
        plan["status"] = "failed_or_cancelled"
        raise
    finally:
        plan["elapsed_seconds"] = round(time.monotonic() - started, 3)
        manifest.write_text(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
