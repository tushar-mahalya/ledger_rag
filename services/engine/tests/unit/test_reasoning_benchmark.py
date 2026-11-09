"""Reasoning benchmarks: raw vs Ledger arms, offline replay, sealing, registry, privacy."""

import json
from pathlib import Path

import pytest

from ledger_rag.evaluation import models
from ledger_rag.evaluation.benchmarks import reasoning
from ledger_rag.evaluation.official import finqa as official
from ledger_rag.reasoning import direct, typed
from ledger_rag.reasoning.grounding import Evidence
from ledger_rag.reasoning.solver import Problem

REPO = Path(__file__).parents[4]


def _record(index: int, split: str, program: str, answer: float) -> dict[str, object]:
    return {
        "id": f"{split}-{index}",
        "filename": f"ACME/20{10 + index}/page_{index}.pdf",
        "pre_text": ["net revenue was $ 5829 million in 2009 and $ 5735 million in 2008 ."],
        "post_text": ["private narrative that must not reach the report ."],
        "table": [["", "2009", "2008"], ["net revenue", "$ 5829", "$ 5735"]],
        "qa": {
            "question": f"question {index}: what is the change in net revenue?",
            "program": program,
            "exe_ans": answer,
            "gold_inds": {"text_0": "net revenue was $ 5829 million in 2009"},
        },
    }


@pytest.fixture
def finqa_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(reasoning, "verify", lambda directory, dataset: None)
    rows = {
        "train": [_record(i, "train", "subtract(5829, 5735)", 94.0) for i in range(3)],
        "dev": [
            _record(0, "dev", "subtract(5829, 5735)", 94.0),
            _record(1, "dev", "subtract(5829, 5735), divide(#0, 5735)", 0.01639),
            _record(2, "dev", "table_sum(net revenue, none)", 11564.0),
        ],
        "test": [_record(0, "test", "subtract(5829, 5735)", 94.0)],
    }
    for split, records in rows.items():
        (tmp_path / f"{split}.json").write_text(json.dumps(records))
    return tmp_path


def test_replay_scores_both_arms_pairs_them_and_reports_no_raw_text(
    finqa_dir: Path, tmp_path: Path
) -> None:
    output = tmp_path / "report.json"
    argv = ["--data-dir", str(finqa_dir), "--samples", "2", "--output", str(output)]
    assert reasoning.main(argv) == 0
    report = json.loads(output.read_text())
    assert set(report["arms"]) == {"raw", "ledger"}
    for arm in report["arms"].values():
        assert arm["metrics"]["official_accuracy"]["mean"] == 1.0
    assert report["arms"]["ledger"]["consistency_curve"] == {"1": 1.0, "2": 1.0}
    delta = report["paired"]["ledger-minus-raw"]["official_accuracy"]
    assert delta["delta"] == 0.0 and delta["n"] == 3
    first = report["results"][0]["arms"]
    assert first["ledger"]["exemplar_ids"] and not first["raw"]["exemplar_ids"]
    text = output.read_text()
    assert "private narrative" not in text and "question 0" not in text


def test_test_split_is_sealed_and_endpoint_needs_a_model(finqa_dir: Path) -> None:
    with pytest.raises(SystemExit):
        reasoning.main(["--data-dir", str(finqa_dir), "--split", "test"])
    with pytest.raises(SystemExit):
        reasoning.main(["--data-dir", str(finqa_dir), "--reader", "endpoint"])
    with pytest.raises(SystemExit):
        reasoning.main(["--data-dir", str(finqa_dir), "--pipelines", "raw,magic"])


def test_registry_specs_are_valid_and_round_trip() -> None:
    specs = models.load_all(REPO / "config" / "models")
    assert {"qwen3.5-4b", "phi-4-mini", "gemini-3.5-flash-lite"} <= {s.id for s in specs}
    for spec in specs:
        config = spec.generation()
        assert config.model == spec.served_name and config.name == f"{spec.id}:{spec.default_mode}"
        assert models.ModelSpec.from_manifest(spec.manifest()) == spec
    qwen = models.load("qwen3.5-4b", REPO / "config" / "models")
    assert qwen.generation().extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    with pytest.raises(ValueError):
        qwen.generation("turbo")


@pytest.mark.parametrize(
    "data",
    [
        {"id": "x", "serving": {"kind": "vllm", "repo": "a/b", "revision": "main", "gpu": "L4"}},
        {"id": "x", "serving": {"kind": "vllm", "repo": "a/b", "revision": "a" * 40, "gpu": "Z"}},
        {"id": "Bad Id", "serving": {"kind": "hosted"}},
        {"id": "x", "serving": {"kind": "hosted", "model": "m"}},
    ],
)
def test_registry_rejects_unpinned_or_incomplete_specs(data: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        models.parse({**data, "modes": {"off": {}}})


def test_raw_answers_parse_leniently_and_lenient_metric_accepts_percent() -> None:
    assert direct.read_number("$1,234") == 1234.0
    assert direct.read_number("16.4%") == pytest.approx(0.164)
    assert direct.read_number("(5)") == -5.0
    assert direct.read_number("about five") is None
    task = direct.DirectNumberTask()
    problem = Problem("q", "?", [], [], [])
    empty = Evidence(frozenset(), frozenset())
    assert task.judge("so...\nANSWER: 16.4%", problem, empty).value == 0.164
    assert task.judge("ANSWER: Yes.", problem, empty).value == "yes"
    assert task.judge("no answer line", problem, empty).reason == "parse"
    assert official.lenient_correct(16.39, 0.1639) and not official.tolerant_correct(16.39, 0.1639)


def test_typed_contract_grounds_spans_and_runs_programs() -> None:
    problem = Problem(
        "q", "?", [], ["Revenue was driven by cloud services."], [["", "2019"], ["revenue", "120"]]
    )
    evidence = Evidence.build(problem.post_text, problem.table, "")
    task = typed.TypedAnswerTask(reasoning.official_executor)
    span = task.judge("TYPE: span\nSCALE: none\nANSWER: cloud services", problem, evidence)
    assert span.reason == "ok" and typed.decode(str(span.value))["answer"] == ["cloud services"]
    invented = task.judge("TYPE: span\nSCALE: none\nANSWER: mobile games", problem, evidence)
    assert invented.reason == "ungrounded"
    sums = task.judge("TYPE: arithmetic\nSCALE: million\nPROGRAM: add(120, 120)", problem, evidence)
    assert typed.decode(str(sums.value)) == {
        "type": "arithmetic",
        "answer": ["240.00"],
        "scale": "million",
    }
    assert task.judge("TYPE: guess\nSCALE: none\nANSWER: x", problem, evidence).reason == "parse"


@pytest.mark.parametrize(
    ("derivation", "program"),
    [
        ("(16,284 - 6,509) / 6,509", "subtract(16284, 6509), divide(#0, 6509)"),
        ("12,907 - 8,538 ", "subtract(12907, 8538)"),
        ("45.1% - 46.3%", "subtract(45.1, 46.3)"),
        (
            "[(2.9+2.9)/2] - [(2.7+2.7)/2]",
            "add(2.9, 2.9), divide(#0, 2), add(2.7, 2.7), divide(#2, 2), subtract(#1, #3)",
        ),
        ("not arithmetic", None),
    ],
)
def test_derivations_become_programs(derivation: str, program: str | None) -> None:
    assert typed.expression_program(derivation) == program


def test_merge_replaces_an_arm_and_recomputes_paired_deltas(
    finqa_dir: Path, tmp_path: Path
) -> None:
    from ledger_rag.evaluation.benchmarks import reasoning_merge

    base, rerun = tmp_path / "base.json", tmp_path / "rerun.json"
    common = ["--data-dir", str(finqa_dir), "--samples", "1"]
    assert reasoning.main([*common, "--output", str(base)]) == 0
    assert reasoning.main([*common, "--pipelines", "raw", "--output", str(rerun)]) == 0
    report, again = json.loads(base.read_text()), json.loads(rerun.read_text())
    for row in again["results"]:  # Pretend the rerun raw arm missed every answer.
        row["arms"]["raw"]["metrics"] = {k: 0.0 for k in row["arms"]["raw"]["metrics"]}
    merged = reasoning_merge.merge(report, again)
    assert merged["paired"]["ledger-minus-raw"]["official_accuracy"]["delta"] == 1.0
    assert merged["merged"]["replaced_arms"] == ["raw"]
    again["results"] = again["results"][:-1]
    with pytest.raises(ValueError):
        reasoning_merge.merge(report, again)
