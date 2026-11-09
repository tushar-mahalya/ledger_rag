"""TAT-QA exact match and F1 with the official metric (vendored, MIT).

Each call scores one question with a fresh upstream ``TaTQAEmAndF1`` so per-item
scores (for bootstrap intervals) are exactly what the official script accumulates.
Requires the ``scoring`` dependency group (NumPy, SciPy, pandas).
"""

from typing import Any

from ledger_rag.evaluation.official import _tatqa_metric_upstream

# Untyped upstream code is called through ``Any`` so it stays byte-identical.
_UPSTREAM: Any = _tatqa_metric_upstream


def score(gold_question: dict[str, Any], answer: Any, scale: str) -> tuple[float, float]:
    """``(exact_match, f1)`` for one prediction; ``answer`` is a string, number or list."""
    metric = _UPSTREAM.TaTQAEmAndF1()
    metric(ground_truth=gold_question, prediction=answer, pred_scale=scale)
    exact, f1, _, _ = metric.get_overall_metric()
    return float(exact), float(f1)
