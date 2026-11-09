"""Replace arms of a reasoning report with the same arms re-run on identical questions.

Used when one arm must be re-measured (for example after an infrastructure fault)
without paying to regenerate the others. The reports must share benchmark, split,
setting, model, data and selection lineage and the exact question order; the
replaced arms' summaries come from the rerun and every paired delta is recomputed.

    python -m ledger_rag.evaluation.benchmarks.reasoning_merge base.json rerun.json -o merged.json
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ledger_rag.evaluation.uncertainty import paired_cluster_difference

MATCHING = ("benchmark", "split", "setting")
LINEAGE = ("dataset_commit", "split_sha256", "selection_seed")


def _check(base: dict[str, Any], rerun: dict[str, Any]) -> None:
    for key in MATCHING:
        if base.get(key) != rerun.get(key):
            raise ValueError(f"reports differ in {key}")
    if (base.get("model") or {}).get("id") != (rerun.get("model") or {}).get("id"):
        raise ValueError("reports are for different models")
    if base.get("generation") != rerun.get("generation"):
        raise ValueError("reports used different generation settings")
    for key in LINEAGE:
        if base["lineage"].get(key) != rerun["lineage"].get(key):
            raise ValueError(f"reports differ in lineage {key}")
    if base.get("retrieval") != rerun.get("retrieval"):
        raise ValueError("reports used different retrieval contexts")
    if [r["id"] for r in base["results"]] != [r["id"] for r in rerun["results"]]:
        raise ValueError("reports cover different questions")
    for arm in rerun["arms"]:
        if base["lineage"]["prompt_sha256"].get(arm) != rerun["lineage"]["prompt_sha256"].get(arm):
            raise ValueError(f"the {arm} arm used a different prompt")


def paired(report: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Every non-raw arm minus ``raw``, per metric, from the per-question results."""
    results = report["results"]
    clusters = {row["id"]: row["cluster"] for row in results}
    out: dict[str, Any] = {}
    for arm in report["arms"]:
        if arm == "raw":
            continue
        metrics = results[0]["arms"][arm]["metrics"] if results else {}
        out[f"{arm}-minus-raw"] = {
            metric: paired_cluster_difference(
                {row["id"]: row["arms"]["raw"]["metrics"][metric] for row in results},
                {row["id"]: row["arms"][arm]["metrics"][metric] for row in results},
                clusters,
                seed=seed,
            )
            for metric in metrics
        }
    return out


def merge(base: dict[str, Any], rerun: dict[str, Any]) -> dict[str, Any]:
    _check(base, rerun)
    merged: dict[str, Any] = json.loads(json.dumps(base))
    replaced = sorted(rerun["arms"])
    for row, new in zip(merged["results"], rerun["results"], strict=True):
        for arm in replaced:
            row["arms"][arm] = new["arms"][arm]
    for arm in replaced:
        merged["arms"][arm] = rerun["arms"][arm]
    merged["paired"] = paired(merged, seed=int(merged["lineage"]["selection_seed"]))
    merged["merged"] = {
        "base_run_id": base.get("run_id"),
        "rerun_run_id": rerun.get("run_id"),
        "replaced_arms": replaced,
        "rerun_provider_stats": rerun.get("provider_stats"),
    }
    return merged


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("base", type=Path)
    parser.add_argument("rerun", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)
    merged = merge(json.loads(args.base.read_text()), json.loads(args.rerun.read_text()))
    args.output.write_text(json.dumps(merged, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
