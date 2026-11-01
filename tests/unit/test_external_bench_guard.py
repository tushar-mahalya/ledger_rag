from argparse import Namespace
from pathlib import Path

import pytest

from ledger_rag.evaluation.benchmarks.finqa import bench_settings
from ledger_rag.evaluation.harness import evaluate


def test_finqa_ignores_ambient_external_enablement_and_rejects_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LEDGER_EXTERNAL_DATA_ENABLED", "true")
    args = Namespace(
        qdrant_url="http://localhost:6333",
        collection="benchmark",
        embedding_url="http://localhost:8000",
        embedding_key="offline-test-token",
        embedding_model="test",
        embedding_space_id="test",
        embedding_dimensions=32,
        query_prefix="",
        quantization="none",
        oversampling=1,
        rerank_candidates=10,
    )
    assert bench_settings(args).external_data_enabled is False
    with pytest.raises(ValueError, match="closed-corpus"):
        bench_settings(args, external_data_enabled=True)


async def test_evaluation_refuses_before_reading_corpus_or_creating_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LEDGER_EXTERNAL_DATA_ENABLED", "true")
    with pytest.raises(ValueError, match="closed-corpus"):
        await evaluate(Path("does-not-exist.jsonl"), Path("does-not-exist-cases.jsonl"))
