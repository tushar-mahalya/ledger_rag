from datetime import date
from pathlib import Path

import pytest

from ledger_rag.core.models import Document
from ledger_rag.evaluation.benchmarks.finqa import Case, Corpus
from ledger_rag.training.data import (
    BM25,
    build_ir_set,
    build_triplets,
    fingerprint,
    is_false_negative,
)
from ledger_rag.training.registry import (
    ModelManifest,
    PromotionPolicy,
    allocate_version,
    decide,
    next_version,
    promoted,
    strip_prefix,
    write_manifest,
)


def _doc(doc_id: str, text: str) -> Document:
    company, year = doc_id.split("/")[:2]
    return Document(
        id=doc_id,
        tenant_id="finqa",
        title=f"{company} annual report {year}",
        text=text,
        source="test",
        published_at=date(int(year), 12, 31),
        allowed_groups=[f"company:{company}"],
    )


def _corpus() -> Corpus:
    corpus = Corpus()
    pages = {
        "AAA/2019/page_1.pdf": "interest expense was $ 123.4 million in 2019 .",
        "AAA/2018/page_2.pdf": "interest expense was $ 123.4 million in 2019 and 98.7 in 2018 .",
        "AAA/2017/page_3.pdf": "revenue grew on interest income and new products .",
        "AAA/2016/page_9.pdf": "interest expense commentary for the held out test page .",
        "BBB/2019/page_4.pdf": "the board approved a dividend of 55.5 per share .",
    }
    for doc_id, text in pages.items():
        corpus.documents[doc_id] = _doc(doc_id, text)
        corpus.companies.add(doc_id.split("/")[0])
    return corpus


def _case(doc_id: str = "AAA/2019/page_1.pdf", gold: str = "$ 123.4 million") -> Case:
    return Case(
        id=f"{doc_id}-q",
        question="what was the interest expense in 2019?",
        document_id=doc_id,
        company=doc_id.split("/")[0],
        gold_units=[gold],
        answer=123.4,
        program="",
    )


def test_false_negative_filter_rejects_restated_prior_year_figures() -> None:
    case = _case()
    assert is_false_negative("interest expense was $ 123.4 million in 2019", case)
    assert is_false_negative("2019 figure 123.4 restated", case)
    assert not is_false_negative("revenue grew on interest income", case)


def test_triplets_use_same_company_hard_negatives_and_exclude_held_out_pages() -> None:
    held_out = {"AAA/2016/page_9.pdf"}
    triplets, stats = build_triplets(
        _corpus(), [_case()], excluded_documents=held_out, negatives_per_query=2
    )
    negatives = {t.negative for t in triplets}
    assert stats.queries == 1
    assert stats.filtered_false_negatives >= 1  # 2018 page repeats the 2019 figure
    assert all("held out" not in n for n in negatives)
    assert all("123.4" not in n for n in negatives)
    assert any("AAA annual report 2017" in n for n in negatives)
    assert all(t.positive.startswith("AAA annual report 2019\n") for t in triplets)
    assert all(t.anchor == "what was the interest expense in 2019?" for t in triplets)


def test_training_on_a_held_out_page_is_rejected() -> None:
    with pytest.raises(ValueError, match="held-out"):
        build_triplets(_corpus(), [_case()], excluded_documents={"AAA/2019/page_1.pdf"}, seed=1)


def test_cases_without_gold_evidence_are_skipped_not_guessed() -> None:
    triplets, stats = build_triplets(
        _corpus(), [_case(gold="not on the page")], excluded_documents=set()
    )
    assert triplets == [] and stats.skipped_without_gold == 1


def test_mining_is_deterministic_and_fingerprinted() -> None:
    args = {"excluded_documents": set(), "negatives_per_query": 3, "seed": 5}
    first, _ = build_triplets(_corpus(), [_case()], **args)
    second, _ = build_triplets(_corpus(), [_case()], **args)
    assert first == second
    assert fingerprint(first) == fingerprint(second)
    assert fingerprint(first) != fingerprint(first[::-1])


def test_ir_set_covers_full_corpus_with_gold_relevance() -> None:
    ir = build_ir_set(_corpus(), [_case(), _case(gold="absent")])
    assert len(ir.corpus) == 5
    assert list(ir.queries) == ["AAA/2019/page_1.pdf-q"]
    (relevant,) = ir.relevant.values()
    assert all("123.4" in ir.corpus[c] for c in relevant)


def test_bm25_ranks_term_overlap_first() -> None:
    scores = BM25([["interest", "expense"], ["dividend"], []]).scores(["interest"])
    assert scores[0] > scores[1] == scores[2] == 0


def test_promotion_requires_min_delta_and_no_guard_regression() -> None:
    policy = PromotionPolicy(min_delta=0.01, guard_metrics=["recall"], max_regression=0.005)
    base = {"cosine_ndcg@10": 0.50, "recall": 0.80}
    assert decide({"cosine_ndcg@10": 0.52, "recall": 0.80}, base, policy).promote
    assert not decide({"cosine_ndcg@10": 0.505, "recall": 0.80}, base, policy).promote
    regressed = decide({"cosine_ndcg@10": 0.60, "recall": 0.79}, base, policy)
    assert not regressed.promote and any("guard recall" in r for r in regressed.reasons)
    assert not decide({"recall": 0.9}, base, policy).promote


def test_versions_are_allocated_atomically_and_pointer_tracks_promotion(tmp_path: Path) -> None:
    assert next_version(["v0001", "v0009", "junk"]) == "v0010"
    v1, _ = allocate_version(tmp_path, "emb")
    v2, _ = allocate_version(tmp_path, "emb")
    assert (v1, v2) == ("v0001", "v0002")
    decision = decide(
        {"cosine_ndcg@10": 0.6}, {"cosine_ndcg@10": 0.5}, PromotionPolicy(guard_metrics=[])
    )
    manifest = ModelManifest(
        name="emb",
        version=v2,
        kind="embedder",
        base_model="base",
        git_sha="abc",
        data_fingerprint="f",
        hyperparameters={"epochs": 1},
        hardware="H100",
        train_seconds=1.0,
        baseline={"cosine_ndcg@10": 0.5},
        candidate={"cosine_ndcg@10": 0.6},
        decision=decision,
    )
    write_manifest(tmp_path, manifest)
    assert promoted(tmp_path, "emb") == "v0002"
    rejected = manifest.model_copy(
        update={"version": v1, "decision": decision.model_copy(update={"promote": False})}
    )
    write_manifest(tmp_path, rejected)
    assert promoted(tmp_path, "emb") == "v0002"


def test_strip_prefix_only_keeps_named_evaluator_metrics() -> None:
    assert strip_prefix({"dev_ndcg@10": 0.5, "test_ndcg@10": 0.1}, "dev") == {"ndcg@10": 0.5}
