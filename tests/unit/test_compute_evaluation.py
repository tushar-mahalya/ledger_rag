import ast
import importlib.util
import json
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


def test_evaluation_job_passes_self_hosted_arms_without_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _load("compute/evaluation.py", "local_evaluation_job_self_hosted")
    monkeypatch.setenv("LEDGER_QDRANT_URL", "https://qdrant.test")
    monkeypatch.setenv("LEDGER_LLM_API_KEY", "serve-key")
    arms = [
        {
            "label": "q:nothink",
            "model": "q",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    ]
    options = {
        "llm_endpoint_kind": "self-hosted",
        "llm_base_url": "https://ws--ledger-llm-serve-serve.modal.run/v1",
        "arms": arms,
        "llm_model_revision": "0" * 40,
        "llm_concurrency": 64,
        "answer_sample_strategy": "stratified",
        "llm_deadline": 600.0,
    }
    parser = build_parser()
    argv = job.benchmark_argv("answers", options, "data", "k", "/data/evaluation/x.json")
    assert "serve-key" not in " ".join(argv)
    args = parser.parse_args(argv)
    validate_args(parser, args)
    assert args.llm_endpoint_kind == "self-hosted" and args.llm_concurrency == 64
    assert args.llm_url == options["llm_base_url"] and args.llm_key == "serve-key"
    assert json.loads(args.llm_arms) == arms and args.llm_deadline == 600
    assert args.answer_sample_strategy == "stratified" and args.llm_rpm is None
    with pytest.raises(ValueError, match="llm_endpoint_kind"):
        job.benchmark_argv("answers", {"llm_endpoint_kind": "cloud"}, "data", "k")
    assert job.LLM_SERVE_APP == "ledger-llm-serve"


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
