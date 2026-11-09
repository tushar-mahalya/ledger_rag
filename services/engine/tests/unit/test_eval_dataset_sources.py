from pathlib import Path

import pytest

from ledger_rag.evaluation.datasets.fetch import fetch_artifact, sha256_file, verify
from ledger_rag.evaluation.datasets.schema import format_number
from ledger_rag.evaluation.datasets.sources import SOURCES, Artifact


def test_format_number_is_compact_and_stable() -> None:
    assert format_number(120.0) == "120"
    assert format_number(-0.21053) == "-0.21053"
    with pytest.raises(TypeError):
        format_number(True)


def test_fetch_verifies_checksums_and_never_keeps_bad_downloads(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream.json"
    upstream.write_text('{"ok": true}')
    good = Artifact("data.json", upstream.as_uri(), sha256_file(upstream))
    destination = tmp_path / "dest"
    assert fetch_artifact(good, destination) is True
    assert fetch_artifact(good, destination) is False
    bad = Artifact("bad.json", upstream.as_uri(), "0" * 64)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        fetch_artifact(bad, destination)
    assert sorted(path.name for path in destination.iterdir()) == ["data.json"]


def test_verify_requires_pinned_artifacts(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify(tmp_path, "financebench")
    (tmp_path / "financebench_merged.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify(tmp_path, "financebench")


def test_sources_are_pinned_and_financebench_is_eval_only() -> None:
    for source in SOURCES.values():
        assert len(source.revision) == 40
        for artifact in source.artifacts:
            assert source.revision in artifact.url and len(artifact.sha256) == 64
    assert SOURCES["financebench"].usage == "eval_only"
    assert {name: s.license for name, s in SOURCES.items()} == {
        "finqa": "MIT",
        "convfinqa": "MIT",
        "tatqa": "CC-BY-4.0",
        "financebench": "CC-BY-NC-4.0",
    }


def test_dataset_cli_rejects_unknown_datasets() -> None:
    from ledger_rag.evaluation.datasets.cli import main

    with pytest.raises(SystemExit):
        main(["fetch", "not-a-dataset"])
    with pytest.raises(SystemExit):
        main(["build", "finqa"])  # consumed raw; nothing to normalize


def test_finqa_is_pinned_like_every_other_dataset() -> None:
    finqa = SOURCES["finqa"]
    assert [a.filename for a in finqa.artifacts] == ["train.json", "dev.json", "test.json"]
    assert all(finqa.revision in a.url for a in finqa.artifacts)
