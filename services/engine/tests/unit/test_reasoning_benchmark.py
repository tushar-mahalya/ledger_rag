"""FinQA program-of-thought runner: offline replay, sealing and report privacy."""

import json
from pathlib import Path

import pytest

from ledger_rag.evaluation.benchmarks import finqa_reasoning


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
    monkeypatch.setattr(finqa_reasoning, "verify", lambda directory, dataset: None)
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


def test_replay_gold_scores_one_and_reports_no_raw_text(finqa_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    argv = ["--data-dir", str(finqa_dir), "--samples", "2", "--output", str(output)]
    assert finqa_reasoning.main(argv) == 0
    report = json.loads(output.read_text())
    metrics = report["summary"]["metrics"]
    assert metrics["official_accuracy"]["mean"] == 1.0
    assert metrics["tolerant_accuracy"]["mean"] == 1.0
    assert report["summary"]["consistency_curve"] == {"1": 1.0, "2": 1.0}
    assert report["setting"] == "gold-page" and report["questions"] == 3
    assert report["results"][0]["exemplar_ids"]  # Dynamic exemplars came from train.
    text = output.read_text()
    assert "private narrative" not in text and "question 0" not in text


def test_test_split_is_sealed(finqa_dir: Path) -> None:
    with pytest.raises(SystemExit):
        finqa_reasoning.main(["--data-dir", str(finqa_dir), "--split", "test"])


def test_endpoint_reader_needs_a_model(finqa_dir: Path) -> None:
    with pytest.raises(SystemExit):
        finqa_reasoning.main(["--data-dir", str(finqa_dir), "--reader", "endpoint"])
