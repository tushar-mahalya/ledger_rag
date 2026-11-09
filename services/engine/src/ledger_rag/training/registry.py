"""File-based model registry with an explicit, auditable promotion gate.

Layout (on a Modal Volume or any filesystem)::

    <root>/<name>/<version>/            model weights (+ onnx/model.onnx for CPU serving)
    <root>/<name>/<version>/manifest.json
    <root>/<name>/PROMOTED              text file containing the promoted version

A candidate is promoted only when it beats the *baseline evaluated on the same
dev set in the same job* by ``min_delta`` on the primary metric and no guard metric
regresses by more than ``max_regression``. The test split is evaluated only after
the decision, so it never influences selection.
"""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

_VERSION = re.compile(r"^v(\d{4})$")


class PromotionPolicy(BaseModel):
    metric: str = "cosine_ndcg@10"
    min_delta: float = Field(default=0.01, ge=0)
    guard_metrics: list[str] = Field(default_factory=lambda: ["cosine_recall@10", "cosine_mrr@10"])
    max_regression: float = Field(default=0.005, ge=0)


class PromotionDecision(BaseModel):
    promote: bool
    reasons: list[str]


class ModelManifest(BaseModel):
    name: str
    version: str
    kind: Literal["embedder", "reranker"]
    base_model: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    git_sha: str
    data_fingerprint: str
    hyperparameters: dict[str, float | int | str | bool | list[int]]
    hardware: str
    train_seconds: float
    baseline: dict[str, float]
    candidate: dict[str, float]
    baseline_test: dict[str, float] | None = None
    test: dict[str, float] | None = None
    int8_dev: dict[str, float] | None = None
    decision: PromotionDecision
    onnx_int8: bool = False


def next_version(existing: list[str]) -> str:
    numbers = [int(m.group(1)) for v in existing if (m := _VERSION.match(v))]
    return f"v{max(numbers, default=0) + 1:04d}"


def decide(
    candidate: dict[str, float], baseline: dict[str, float], policy: PromotionPolicy
) -> PromotionDecision:
    reasons: list[str] = []
    for metric in [policy.metric, *policy.guard_metrics]:
        if metric not in candidate or metric not in baseline:
            return PromotionDecision(promote=False, reasons=[f"missing metric {metric}"])
    delta = candidate[policy.metric] - baseline[policy.metric]
    ok = delta >= policy.min_delta
    reasons.append(
        f"{policy.metric} {baseline[policy.metric]:.4f} -> {candidate[policy.metric]:.4f} "
        f"(delta {delta:+.4f}, required >= {policy.min_delta:+.4f})"
    )
    for metric in policy.guard_metrics:
        drop = baseline[metric] - candidate[metric]
        if drop > policy.max_regression:
            ok = False
            reasons.append(f"guard {metric} regressed by {drop:.4f} > {policy.max_regression}")
    return PromotionDecision(promote=ok, reasons=reasons)


def versions(root: Path, name: str) -> list[str]:
    base = root / name
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir() and _VERSION.match(p.name))


def allocate_version(root: Path, name: str) -> tuple[str, Path]:
    """Atomically reserve the next version directory (safe against concurrent jobs)."""
    for _ in range(100):
        version = next_version(versions(root, name))
        path = root / name / version
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return version, path
    raise RuntimeError(f"Could not allocate a version for {name}")


def strip_prefix(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    """``{"dev_cosine_ndcg@10": x}`` -> ``{"cosine_ndcg@10": x}`` (evaluator name prefix)."""
    head = f"{prefix}_"
    return {k.removeprefix(head): float(v) for k, v in metrics.items() if k.startswith(head)}


def write_manifest(root: Path, manifest: ModelManifest) -> Path:
    path = root / manifest.name / manifest.version / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(manifest.model_dump_json(indent=2))
    tmp.replace(path)
    if manifest.decision.promote:
        pointer = root / manifest.name / "PROMOTED"
        tmp = pointer.with_suffix(".tmp")
        tmp.write_text(manifest.version)
        tmp.replace(pointer)
    return path


def promoted(root: Path, name: str) -> str | None:
    pointer = root / name / "PROMOTED"
    return pointer.read_text().strip() if pointer.exists() else None
