"""Ledger RAG compute plane: retriever fine-tuning, evaluation, export and promotion on Modal.

Stages (each a Modal Function with right-sized hardware):

    fetch_finqa      CPU   pinned FinQA download + SHA-256 verification -> ledger-data volume
                           (the shared registry in ledger_rag.evaluation.datasets)
    build_dataset    CPU   leakage-safe triplets/IR sets (ledger_rag.training.data)
    finetune_*       GPU   baseline eval -> train -> candidate eval (dev) -> promotion gate
                           -> held-out test eval -> ONNX fp32 + INT8 export -> registry
    evaluate_int8    CPU   measures the INT8 quantization delta on dev (what CPU TEI serves)

Usage::

    uv run --group compute modal run compute/training.py --kind embedder
    uv run --group compute modal run compute/training.py --kind reranker

The container environment is built from this repository's ``uv.lock`` (``train`` group),
so local, CI and remote runs resolve identical dependency versions.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "ledger-training"
DATA, MODELS, CACHE = Path("/data"), Path("/models"), Path("/cache")
# Fine-tuning a ~150M-parameter encoder takes minutes; H100 finishes fastest at modest cost.
TRAIN_GPU = os.environ.get("LEDGER_MODAL_TRAIN_GPU", "H100")
EMBEDDER_BASE = "Alibaba-NLP/gte-modernbert-base"
RERANKER_BASE = "Alibaba-NLP/gte-reranker-modernbert-base"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "git")
    # Hash-pinned export of uv.lock (scripts/export_requirements.py); uv_sync does not
    # support uv workspaces.
    .uv_pip_install(
        requirements=[str(Path(__file__).with_name("requirements") / "engine-train.txt")]
    )
    .env({"HF_HOME": str(CACHE / "hf"), "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("ledger_rag")
)

app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name("ledger-data", create_if_missing=True)
models_volume = modal.Volume.from_name("ledger-models", create_if_missing=True)
cache_volume = modal.Volume.from_name("ledger-hf-cache", create_if_missing=True)
VOLUMES: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    str(DATA): data_volume,
    str(MODELS): models_volume,
    str(CACHE): cache_volume,
}
RETRIES = modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=5.0)


def _log(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, "ts": round(time.time(), 3), **fields}, default=str))


@app.function(volumes=VOLUMES, timeout=900, retries=RETRIES)
def fetch_finqa() -> str:
    target = DATA / "finqa"
    from ledger_rag.evaluation.datasets.fetch import fetch_verified

    fetch_verified("finqa", target)
    data_volume.commit()
    _log("finqa.fetched", path=str(target))
    return str(target)


@app.function(volumes=VOLUMES, cpu=4, memory=8192, timeout=1800)
def build_dataset(
    seed: int = 13, negatives: int = 3, rerank_candidates: int = 19
) -> dict[str, Any]:
    from dataclasses import asdict

    from ledger_rag.evaluation.benchmarks.finqa import load
    from ledger_rag.training.data import build_ir_set, build_triplets, fingerprint

    data_volume.reload()
    source = DATA / "finqa"
    corpus, train = load(source, "train")
    _, dev = load(source, "dev")
    _, test = load(source, "test")
    held_out = {c.document_id for c in test}
    triplets, stats = build_triplets(
        corpus, train, excluded_documents=held_out, negatives_per_query=negatives, seed=seed
    )
    fp = fingerprint(triplets)
    out = DATA / "datasets" / f"finqa-{fp[:12]}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "train.jsonl").open("w") as fh:
        for row in triplets:
            fh.write(json.dumps(asdict(row)) + "\n")
    for split, cases in {"dev": dev, "test": test}.items():
        ir = build_ir_set(corpus, cases)
        (out / f"{split}_ir.json").write_text(
            json.dumps(
                {
                    "queries": ir.queries,
                    "corpus": ir.corpus,
                    "relevant": {k: sorted(v) for k, v in ir.relevant.items()},
                }
            )
        )
        # Reranking task: gold passages vs mined same-company hard negatives.
        exclude = held_out if split == "dev" else set()
        rerank, _ = build_triplets(
            corpus,
            cases,
            excluded_documents=exclude,
            negatives_per_query=rerank_candidates,
            positives_per_query=3,
            seed=seed,
        )
        samples: dict[str, dict[str, Any]] = {}
        for t in rerank:
            sample = samples.setdefault(
                t.query_id, {"query": t.anchor, "positive": [], "negative": []}
            )
            if t.positive not in sample["positive"]:
                sample["positive"].append(t.positive)
            if t.negative not in sample["negative"]:
                sample["negative"].append(t.negative)
        (out / f"{split}_rerank.json").write_text(json.dumps(list(samples.values())))
    summary = {
        "path": str(out),
        "fingerprint": fp,
        "seed": seed,
        "triplets": len(triplets),
        "mining": asdict(stats),
        "train_queries": len(train),
        "dev_queries": len(dev),
        "test_queries": len(test),
    }
    (out / "stats.json").write_text(json.dumps(summary, indent=2))
    data_volume.commit()
    _log("dataset.built", **summary)
    return summary


def _ir_evaluator(path: Path, name: str, truncate_dim: int | None = None) -> Any:
    from sentence_transformers.sentence_transformer.evaluation import (
        InformationRetrievalEvaluator,
    )

    payload = json.loads(path.read_text())
    return InformationRetrievalEvaluator(
        queries=payload["queries"],
        corpus=payload["corpus"],
        relevant_docs={k: set(v) for k, v in payload["relevant"].items()},
        name=name,
        batch_size=256,
        truncate_dim=truncate_dim,
        accuracy_at_k=[1, 3, 5, 10],
        precision_recall_at_k=[1, 3, 5, 10, 20],
        mrr_at_k=[10],
        ndcg_at_k=[10],
        map_at_k=[100],
        show_progress_bar=False,
    )


def _export_onnx(model_cls: Any, path: Path) -> bool:
    """Export fp32 ONNX and an AVX512-VNNI INT8 variant; INT8 becomes onnx/model.onnx.

    TEI's CPU (ORT) backend loads ``onnx/model.onnx``; GPU serving uses safetensors.
    """
    from sentence_transformers import export_dynamic_quantized_onnx_model

    try:
        onnx_model = model_cls(str(path), backend="onnx")
        onnx_model.save_pretrained(str(path))
        export_dynamic_quantized_onnx_model(onnx_model, "avx512_vnni", str(path))
        onnx_dir = path / "onnx"
        (onnx_dir / "model.onnx").replace(onnx_dir / "model_fp32.onnx")
        (onnx_dir / "model_qint8_avx512_vnni.onnx").replace(onnx_dir / "model.onnx")
        return True
    except Exception as exc:  # export is best-effort; the safetensors model remains servable
        _log("onnx.export_failed", error=repr(exc))
        return False


def _lineage_warning(git_sha: str) -> None:
    if not git_sha or git_sha.endswith("-dirty"):
        _log("lineage.warning", detail="training from an uncommitted tree", git_sha=git_sha)


@app.function(gpu=TRAIN_GPU, volumes=VOLUMES, timeout=3 * 3600, memory=32768, cpu=8)
def finetune_embedder(
    dataset: str,
    git_sha: str,
    data_fingerprint: str,
    base_model: str = EMBEDDER_BASE,
    epochs: int = 1,
    batch_size: int = 256,
    learning_rate: float = 2e-5,
    matryoshka_dims: list[int] | None = None,
    seed: int = 13,
    policy: dict[str, Any] | None = None,
    unseal_test: bool = False,
) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.base.sampler import BatchSamplers
    from sentence_transformers.sentence_transformer.losses import (
        CachedMultipleNegativesRankingLoss,
        MatryoshkaLoss,
    )

    from ledger_rag.training.registry import (
        ModelManifest,
        PromotionPolicy,
        allocate_version,
        decide,
        strip_prefix,
        write_manifest,
    )

    _lineage_warning(git_sha)
    data_volume.reload()
    models_volume.reload()
    dims = matryoshka_dims or [768, 512, 256, 128]
    root = Path(dataset)
    rows = [json.loads(line) for line in (root / "train.jsonl").open()]
    train = Dataset.from_dict(
        {
            "anchor": [r["anchor"] for r in rows],
            "positive": [r["positive"] for r in rows],
            "negative": [r["negative"] for r in rows],
        }
    ).shuffle(seed=seed)
    dev = _ir_evaluator(root / "dev_ir.json", "dev")
    dev_small = _ir_evaluator(root / "dev_ir.json", "dev256", truncate_dim=256)

    model = SentenceTransformer(base_model)
    baseline = strip_prefix(dev(model), "dev")
    baseline |= {f"d256_{k}": v for k, v in strip_prefix(dev_small(model), "dev256").items()}
    _log("embedder.baseline", ndcg10=baseline["cosine_ndcg@10"])

    version, out = allocate_version(MODELS, "ledger-embedder")
    loss = MatryoshkaLoss(
        model, CachedMultipleNegativesRankingLoss(model, mini_batch_size=64), dims
    )
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=0.25,
        save_strategy="steps",
        save_steps=0.25,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_dev_cosine_ndcg@10",
        logging_steps=10,
        report_to="none",
        seed=seed,
    )
    started = time.perf_counter()
    SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train, loss=loss, evaluator=dev
    ).train()
    train_seconds = time.perf_counter() - started

    candidate = strip_prefix(dev(model), "dev")
    candidate |= {f"d256_{k}": v for k, v in strip_prefix(dev_small(model), "dev256").items()}
    decision = decide(candidate, baseline, PromotionPolicy(**(policy or {})))
    _log("embedder.decision", promote=decision.promote, reasons=decision.reasons)

    # Test stays sealed unless explicitly unsealed; then it is scored once, after the
    # decision, for reporting only.
    candidate_test = baseline_test = None
    _log("embedder.test_split", unsealed=unseal_test)
    if unseal_test:
        test = _ir_evaluator(root / "test_ir.json", "test")
        candidate_test = strip_prefix(test(model), "test")
        baseline_test = strip_prefix(test(SentenceTransformer(base_model)), "test")

    model.save_pretrained(str(out))
    subprocess.run(["rm", "-rf", str(out / "checkpoints")], check=True)
    onnx = _export_onnx(SentenceTransformer, out)
    manifest = ModelManifest(
        name="ledger-embedder",
        version=version,
        kind="embedder",
        base_model=base_model,
        git_sha=git_sha,
        data_fingerprint=data_fingerprint,
        hyperparameters={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "matryoshka_dims": dims,
            "loss": "MatryoshkaLoss(CachedMNRL)",
            "seed": seed,
        },
        hardware=torch.cuda.get_device_name(0),
        train_seconds=round(train_seconds, 1),
        baseline=baseline,
        candidate=candidate,
        baseline_test=baseline_test,
        test=candidate_test,
        decision=decision,
        onnx_int8=onnx,
    )
    write_manifest(MODELS, manifest)
    models_volume.commit()
    return manifest.model_dump(mode="json")


@app.function(gpu=TRAIN_GPU, volumes=VOLUMES, timeout=3 * 3600, memory=32768, cpu=8)
def finetune_reranker(
    dataset: str,
    git_sha: str,
    data_fingerprint: str,
    base_model: str = RERANKER_BASE,
    epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 2e-5,
    seed: int = 13,
    policy: dict[str, Any] | None = None,
    unseal_test: bool = False,
) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import (
        CrossEncoder,
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    from ledger_rag.training.registry import (
        ModelManifest,
        PromotionPolicy,
        allocate_version,
        decide,
        strip_prefix,
        write_manifest,
    )

    _lineage_warning(git_sha)
    data_volume.reload()
    models_volume.reload()
    root = Path(dataset)
    queries, passages, labels = [], [], []
    seen: set[tuple[str, str]] = set()
    for line in (root / "train.jsonl").open():
        row = json.loads(line)
        for passage, label in ((row["positive"], 1.0), (row["negative"], 0.0)):
            if (row["anchor"], passage) not in seen:
                seen.add((row["anchor"], passage))
                queries.append(row["anchor"])
                passages.append(passage)
                labels.append(label)
    positives = sum(labels)
    train = Dataset.from_dict({"query": queries, "passage": passages, "label": labels}).shuffle(
        seed=seed
    )

    def evaluator(split: str) -> Any:
        return CrossEncoderRerankingEvaluator(
            samples=json.loads((root / f"{split}_rerank.json").read_text()),
            at_k=10,
            name=split,
            batch_size=128,
            always_rerank_positives=True,
        )

    dev = evaluator("dev")
    model = CrossEncoder(base_model)
    baseline = strip_prefix(dev(model), "dev")
    version, out = allocate_version(MODELS, "ledger-reranker")
    loss = BinaryCrossEntropyLoss(
        model, pos_weight=torch.tensor((len(labels) - positives) / max(positives, 1.0))
    )
    args = CrossEncoderTrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        bf16=True,
        eval_strategy="steps",
        eval_steps=0.25,
        save_strategy="steps",
        save_steps=0.25,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_dev_ndcg@10",
        logging_steps=20,
        report_to="none",
        seed=seed,
    )
    started = time.perf_counter()
    CrossEncoderTrainer(
        model=model, args=args, train_dataset=train, loss=loss, evaluator=dev
    ).train()
    train_seconds = time.perf_counter() - started
    candidate = strip_prefix(dev(model), "dev")
    reranker_policy = {"metric": "ndcg@10", "guard_metrics": ["mrr@10", "map"], **(policy or {})}
    decision = decide(candidate, baseline, PromotionPolicy(**reranker_policy))
    _log("reranker.decision", promote=decision.promote, reasons=decision.reasons)
    candidate_test = baseline_test = None
    _log("reranker.test_split", unsealed=unseal_test)
    if unseal_test:
        test = evaluator("test")
        candidate_test = strip_prefix(test(model), "test")
        baseline_test = strip_prefix(test(CrossEncoder(base_model)), "test")
    model.save_pretrained(str(out))
    subprocess.run(["rm", "-rf", str(out / "checkpoints")], check=True)
    onnx = _export_onnx(CrossEncoder, out)
    manifest = ModelManifest(
        name="ledger-reranker",
        version=version,
        kind="reranker",
        base_model=base_model,
        git_sha=git_sha,
        data_fingerprint=data_fingerprint,
        hyperparameters={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "loss": "BinaryCrossEntropy(pos_weight)",
            "seed": seed,
        },
        hardware=torch.cuda.get_device_name(0),
        train_seconds=round(train_seconds, 1),
        baseline=baseline,
        candidate=candidate,
        baseline_test=baseline_test,
        test=candidate_test,
        decision=decision,
        onnx_int8=onnx,
    )
    write_manifest(MODELS, manifest)
    models_volume.commit()
    return manifest.model_dump(mode="json")


@app.function(volumes=VOLUMES, cpu=16, memory=16384, timeout=3600)
def evaluate_int8(name: str, version: str, dataset: str) -> dict[str, float]:
    """Dev metrics of the INT8 ONNX artifact, i.e. exactly what CPU TEI serves."""
    from ledger_rag.training.registry import ModelManifest, strip_prefix, write_manifest

    models_volume.reload()
    path = MODELS / name / version
    root = Path(dataset)
    if name == "ledger-embedder":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(path), backend="onnx")
        metrics = strip_prefix(_ir_evaluator(root / "dev_ir.json", "dev")(model), "dev")
    else:
        from sentence_transformers.cross_encoder import CrossEncoder
        from sentence_transformers.cross_encoder.evaluation import (
            CrossEncoderRerankingEvaluator,
        )

        model = CrossEncoder(str(path), backend="onnx")
        samples = json.loads((root / "dev_rerank.json").read_text())
        metrics = strip_prefix(
            CrossEncoderRerankingEvaluator(samples=samples, at_k=10, name="dev")(model), "dev"
        )
    manifest = ModelManifest.model_validate_json((path / "manifest.json").read_text())
    manifest.int8_dev = metrics
    write_manifest(MODELS, manifest)
    models_volume.commit()
    return metrics


def _git_sha() -> str:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


@app.local_entrypoint()
def main(
    kind: str = "embedder",
    epochs: int = 1,
    batch_size: int = 0,
    learning_rate: float = 2e-5,
    seed: int = 13,
    skip_int8_eval: bool = False,
    unseal_test: bool = False,
) -> None:
    """Train and promote on dev; ``--unseal-test`` additionally reports held-out test."""
    if kind not in {"embedder", "reranker"}:
        raise SystemExit("--kind must be embedder or reranker")
    git_sha = _git_sha()
    fetch_finqa.remote()
    dataset = build_dataset.remote(seed=seed)
    common: dict[str, Any] = {
        "dataset": dataset["path"],
        "git_sha": git_sha,
        "data_fingerprint": dataset["fingerprint"],
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        "unseal_test": unseal_test,
    }
    if unseal_test:
        print("WARNING: test split unsealed; report it once and do not tune on it")
    if kind == "embedder":
        manifest = finetune_embedder.remote(batch_size=batch_size or 256, **common)
    else:
        manifest = finetune_reranker.remote(batch_size=batch_size or 64, **common)
    if manifest["onnx_int8"] and not skip_int8_eval:
        manifest["int8_dev"] = evaluate_int8.remote(
            manifest["name"], manifest["version"], dataset["path"]
        )
    manifest["dataset"] = dataset
    out = ROOT / "benchmarks" / "results" / f"modal-{manifest['name']}-{manifest['version']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("name", "version", "decision")}, indent=2))
    print(f"manifest written to {out.relative_to(ROOT)}")
