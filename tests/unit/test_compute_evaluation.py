import ast
import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

from ledger_rag.evaluation.benchmarks.finqa import bench_settings, build_parser, validate_args

ROOT = Path(__file__).resolve().parents[2]


def _load(path: str, name: str) -> ModuleType:
    pytest.importorskip("modal", reason="Modal job tests require the compute group")
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluation_job_pins_revisions_and_builds_valid_read_only_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _load("compute/evaluation.py", "local_evaluation_job")
    assert re.fullmatch(r"[0-9a-f]{40}", job.EMBEDDING_REVISION)
    assert re.fullmatch(r"[0-9a-f]{40}", job.RERANKER_REVISION)
    assert job.EMBEDDING_SPACE_ID.endswith(f"@{job.EMBEDDING_REVISION}")
    monkeypatch.setenv("LEDGER_QDRANT_URL", "https://qdrant.test")
    parser = build_parser()
    for command in job.COMMANDS:
        args = parser.parse_args(job.benchmark_argv(command, {"limit": 20}, "data", "k"))
        validate_args(parser, args)
        assert args.collection == job.COLLECTION
        assert args.split == "dev" and not args.unseal_test
        assert args.temperature is None  # Gemini 3: keep the provider default
        assert args.expected_corpus_sha256 == job.CORPUS_SHA256
        assert args.reranker_url.startswith("http://127.0.0.1:")
        assert args.reranker_key == args.embedding_key == "k"
        assert bench_settings(args).reranker_api_key is not None
    with pytest.raises(ValueError, match="read-only"):
        job.benchmark_argv("ingest", {}, "data", "k")
    sealed = parser.parse_args(job.benchmark_argv("answers", {"split": "test"}, "data", "k"))
    with pytest.raises(SystemExit):
        validate_args(parser, sealed)
    unsealed = job.benchmark_argv("answers", {"split": "test", "unseal_test": True}, "data", "k")
    validate_args(parser, parser.parse_args(unsealed))


def test_training_keeps_test_split_sealed_by_default() -> None:
    tree = ast.parse((ROOT / "compute/training.py").read_text())
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    for name in ("finetune_embedder", "finetune_reranker", "main"):
        arguments = functions[name].args
        defaults = dict(
            zip(
                [a.arg for a in arguments.args][-len(arguments.defaults) :],
                arguments.defaults,
                strict=True,
            )
        )
        assert isinstance(defaults["unseal_test"], ast.Constant)
        assert defaults["unseal_test"].value is False
    # Test evaluators are only built inside an ``if unseal_test`` branch.
    for name in ("finetune_embedder", "finetune_reranker"):
        guarded = [
            node
            for node in ast.walk(functions[name])
            if isinstance(node, ast.If) and ast.unparse(node.test) == "unseal_test"
        ]
        assert guarded and "test" in ast.unparse(guarded[0])
        outside = ast.unparse(functions[name]).replace(ast.unparse(guarded[0]), "")
        assert '"test")' not in outside
