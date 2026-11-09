"""Render raw-vs-Ledger results as a Markdown table for docs and reviews.

Input: the JSON written by ``compute/reasoning.py --output`` (one entry per model)
or one or more full reports from ``reasoning --output``. Each row is one model on
one benchmark: the primary metric per arm with its 95% CI and the paired
``ledger-minus-raw`` delta. Only aggregate numbers are rendered.

    python -m ledger_rag.evaluation.benchmarks.reasoning_table [--lenient] matrix.json [...]

``--lenient`` scores FinQA-family arms with the percentage-tolerant comparison and
TAT-QA with F1: the fairest view of free-text raw answers.
"""

import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

PRIMARY = {"finqa": "official_accuracy", "convfinqa": "official_accuracy", "tatqa": "exact_match"}
# The fairest comparison for free-text raw answers: percentage-tolerant matching and F1.
LENIENT = {"finqa": "lenient_accuracy", "convfinqa": "lenient_accuracy", "tatqa": "f1"}
ARMS = ("raw", "ledger", "ledger-refine")


def _cell(metric: dict[str, Any] | None) -> str:
    if not metric or metric.get("mean") is None:
        return "—"
    low, high = metric["ci95"]
    return f"{100 * metric['mean']:.1f} [{100 * low:.1f}, {100 * high:.1f}]"


def _delta(delta: dict[str, Any] | None) -> str:
    if not delta or delta.get("delta") is None:
        return "—"
    low, high = delta["ci95"]
    return f"{100 * delta['delta']:+.1f} [{100 * low:+.1f}, {100 * high:+.1f}]"


def _runs(payload: Any) -> Iterator[tuple[str, str, dict[str, Any], dict[str, Any], int | None]]:
    """(model, benchmark, arms metrics, paired deltas, n) from either input shape."""
    entries = payload if isinstance(payload, list) else [payload]
    for entry in entries:
        if "runs" in entry:  # compute/reasoning.py output
            for run in entry["runs"]:
                first: dict[str, Any] = next(iter(run["arms"].values()), {})
                metric: dict[str, Any] = next(iter(first.values()), {})
                n = metric.get("n")
                yield entry["model"], run["benchmark"], run["arms"], run["paired"], n
        elif "arms" in entry:  # a full benchmark report
            arms = {name: arm["metrics"] for name, arm in entry["arms"].items()}
            model = (entry.get("model") or {}).get("id", entry.get("reader", "?"))
            yield model, entry["benchmark"], arms, entry["paired"], entry["questions"]


def table(payloads: Sequence[Any], *, lenient: bool = False) -> str:
    chosen = LENIENT if lenient else PRIMARY
    rows = [
        "| Model | Benchmark (metric) | n | Raw model | Ledger | Ledger + refine | "
        "Ledger − raw (paired) | Ledger + refine − raw |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for payload in payloads:
        for model, benchmark, arms, paired, n in _runs(payload):
            metric = chosen.get(benchmark, "")
            cells = [_cell((arms.get(arm) or {}).get(metric)) for arm in ARMS]
            deltas = [
                _delta((paired.get(f"{arm}-minus-raw") or {}).get(metric)) for arm in ARMS[1:]
            ]
            rows.append(
                f"| {model} | {benchmark} ({metric}) | {n or '—'} | {' | '.join(cells)} | "
                f"{' | '.join(deltas)} |"
            )
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    paths = list(argv if argv is not None else sys.argv[1:])
    lenient = "--lenient" in paths
    paths = [path for path in paths if path != "--lenient"]
    if not paths:
        print(__doc__, file=sys.stderr)
        return 2
    print(table([json.loads(Path(path).read_text()) for path in paths], lenient=lenient))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
