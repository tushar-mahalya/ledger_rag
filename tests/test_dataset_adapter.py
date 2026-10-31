import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_financebench import normalize


def files(tmp_path: Path) -> tuple[Path, Path, Path]:
    source, corpus, mapping = [
        tmp_path / name for name in ["source.jsonl", "corpus.jsonl", "map.json"]
    ]
    corpus.write_text(
        json.dumps({"id": "filing", "text": "Original fictional full disclosure"}) + "\n"
    )
    source.write_text(
        json.dumps(
            {
                "financebench_id": "example",
                "question": "Fictional test question?",
                "doc_name": "fictional",
                "answer": "Gold answer must not be indexed",
                "evidence": [{"evidence_text": "Gold passage"}],
            }
        )
        + "\n"
    )
    mapping.write_text(
        json.dumps(
            {
                "document_ids": {"fictional": "filing"},
                "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
            }
        )
    )
    return source, corpus, mapping


def test_adapter_does_not_index_gold_evidence(tmp_path: Path) -> None:
    source, corpus, mapping = files(tmp_path)
    rows, manifest = normalize(source, corpus, mapping)
    assert rows[0]["relevant_document_ids"] == ["filing"]
    assert "Gold" not in json.dumps(rows)
    assert manifest["rows"] == 1


def test_adapter_requires_independent_corpus(tmp_path: Path) -> None:
    source, corpus, mapping = files(tmp_path)
    corpus.write_text("Changed corpus")
    with pytest.raises(ValueError, match="hash"):
        normalize(source, corpus, mapping)


def test_adapter_rejects_duplicate_cases(tmp_path: Path) -> None:
    source, corpus, mapping = files(tmp_path)
    source.write_text(source.read_text() * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        normalize(source, corpus, mapping)
