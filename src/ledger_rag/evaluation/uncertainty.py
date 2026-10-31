"""Cluster-aware uncertainty, latency and selective-prediction summaries.

FinQA questions share report pages, reports and companies, so items are not
independent. Intervals here resample whole clusters (for FinQA: a report, i.e.
``company/year``) with replacement and recompute the item-weighted mean, which
widens intervals honestly when questions on one filing succeed or fail together.
"""

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def _nearest_rank(ordered: Sequence[float], fraction: float) -> float:
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def cluster_bootstrap(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    seed: int = 7,
    samples: int = 2000,
) -> dict[str, Any]:
    """Mean with a percentile 95% CI from a cluster (block) bootstrap."""
    if len(values) != len(clusters):
        raise ValueError("Each value needs exactly one cluster label")
    if samples < 1 or any(not math.isfinite(v) for v in values):
        raise ValueError("Require positive bootstrap samples and finite values")
    if not values:
        return {"mean": None, "ci95": [None, None], "n": 0, "clusters": 0}
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters, strict=True):
        grouped[cluster].append(value)
    sums = [(sum(group), len(group)) for _, group in sorted(grouped.items())]
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        total = count = 0.0
        for index in rng.choices(range(len(sums)), k=len(sums)):
            total += sums[index][0]
            count += sums[index][1]
        means.append(total / count)
    means.sort()
    return {
        "mean": round(sum(values) / len(values), 4),
        "ci95": [round(_nearest_rank(means, 0.025), 4), round(_nearest_rank(means, 0.975), 4)],
        "n": len(values),
        "clusters": len(sums),
    }


def paired_cluster_difference(
    first: Mapping[str, float],
    second: Mapping[str, float],
    clusters: Mapping[str, str],
    *,
    seed: int = 7,
    samples: int = 2000,
) -> dict[str, Any]:
    """Mean of ``second - first`` over shared item IDs with a paired cluster CI."""
    shared = sorted(set(first) & set(second))
    differences = [second[item] - first[item] for item in shared]
    result = cluster_bootstrap(
        differences, [clusters[item] for item in shared], seed=seed, samples=samples
    )
    return {"delta": result.pop("mean"), **result}


def summarize_rows(
    rows: Sequence[Mapping[str, float | None]],
    clusters: Sequence[str],
    *,
    seed: int = 7,
    samples: int = 2000,
) -> dict[str, Any]:
    """Cluster-bootstrap every metric; ``None`` marks an ineligible item, not a zero."""
    if len(rows) != len(clusters):
        raise ValueError("Each row needs exactly one cluster label")
    names = sorted({name for row in rows for name in row})
    out: dict[str, Any] = {}
    for name in names:
        eligible = [(row[name], cluster) for row, cluster in zip(rows, clusters, strict=True)]
        pairs = [(float(v), c) for v, c in eligible if v is not None]
        out[name] = cluster_bootstrap(
            [v for v, _ in pairs], [c for _, c in pairs], seed=seed, samples=samples
        )
        out[name]["ineligible"] = len(rows) - len(pairs)
    return out


def latency(values: Sequence[float]) -> dict[str, float]:
    """Nearest-rank latency percentiles in milliseconds."""
    ordered = sorted(values)
    if not ordered:
        return {}
    return {
        "n": len(ordered),
        "p50_ms": round(_nearest_rank(ordered, 0.50), 1),
        "p90_ms": round(_nearest_rank(ordered, 0.90), 1),
        "p95_ms": round(_nearest_rank(ordered, 0.95), 1),
        "p99_ms": round(_nearest_rank(ordered, 0.99), 1),
        "mean_ms": round(sum(ordered) / len(ordered), 1),
    }


def risk_coverage(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    points: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> dict[str, Any]:
    """Selective accuracy when answering only the most confident fraction of items.

    Ties are broken by original order (deterministic). AURC is the mean selective
    risk over every coverage prefix; lower is better.
    """
    if len(confidences) != len(correct):
        raise ValueError("Each confidence needs a correctness label")
    if not confidences:
        return {"n": 0, "curve": [], "aurc": None}
    order = sorted(range(len(confidences)), key=lambda i: (-confidences[i], i))
    hits = 0
    risks: list[float] = []
    prefix: list[float] = []
    for rank, index in enumerate(order, 1):
        hits += int(correct[index])
        prefix.append(hits / rank)
        risks.append(1 - hits / rank)
    curve = []
    for point in points:
        count = max(1, math.ceil(point * len(order)))
        curve.append(
            {"coverage": round(count / len(order), 4), "accuracy": round(prefix[count - 1], 4)}
        )
    return {"n": len(order), "curve": curve, "aurc": round(sum(risks) / len(risks), 4)}
