"""Corpus pipeline: specs, gates, versioned builds, promotion, rollback, document admin."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr, ValidationError
from qdrant_client import AsyncQdrantClient

from ledger_rag.core.config import Settings
from ledger_rag.core.models import Document
from ledger_rag.corpus import cli
from ledger_rag.corpus.index import IndexRegistry, PromotionError
from ledger_rag.corpus.pipeline import BuildRejected, build, plan, store_settings
from ledger_rag.corpus.spec import CorpusSpec, load_spec
from ledger_rag.corpus.validate import validate_readback
from ledger_rag.retrieval.qdrant import QdrantRetriever

MODEL = "test-org/test-embedding"
DIMS = 32


def make_spec(**overrides: Any) -> CorpusSpec:
    raw: dict[str, Any] = {
        "id": "filings",
        "description": "Test filings",
        "tenant": "acme",
        "source": {"kind": "jsonl", "path": "filings.jsonl"},
        "access": {"default_groups": [], "classification": "internal"},
        "embedding": {"model": MODEL, "revision": "a" * 40, "dimensions": DIMS},
        "index": {"alias": "filings_idx", "keep_builds": 2},
    }
    raw.update(overrides)
    return CorpusSpec.model_validate(raw)


def document(n: int, tenant: str = "acme", groups: list[str] | None = None) -> Document:
    return Document(
        id=f"doc-{n}",
        tenant_id=tenant,
        title=f"Filing {n}",
        text=f"Revenue for unit {n} was {n * 10} million in fiscal year 2024.\n\nMargins held.",
        source=f"https://filings.test/{n}",
        published_at=date(2024, 6, 1),
        allowed_groups=groups or [],
    )


def embedding_client() -> httpx.AsyncClient:
    def handle(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        rows = [
            {"index": i, "embedding": [1.0 + (len(text) % 7) / 10] * DIMS}
            for i, text in enumerate(texts)
        ]
        return httpx.Response(200, json={"model": MODEL, "data": rows})

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


async def make_store(
    spec: CorpusSpec, collection: str, client: AsyncQdrantClient
) -> QdrantRetriever:
    base = Settings(
        demo_token=SecretStr("demo-test-token"),
        embedding_api_key=SecretStr("embedding-secret"),
        embedding_base_url="https://embedding.test/v1/",
    )
    store = QdrantRetriever(store_settings(base, spec, collection))
    await store._client.close()
    store._client = client
    await store._embedding_client.aclose()
    store._embedding_client = embedding_client()
    return store


@pytest.fixture
def client() -> AsyncQdrantClient:
    return AsyncQdrantClient(":memory:")


async def run_build(
    spec: CorpusSpec, client: AsyncQdrantClient, build_id: str, docs: list[Document], **kw: Any
) -> dict[str, Any]:
    store = await make_store(spec, spec.build_collection(build_id), client)
    return await build(
        plan(spec, docs), store, build_id=build_id, registry=IndexRegistry(client), **kw
    )


def test_spec_rules() -> None:
    with pytest.raises(ValidationError):
        make_spec(source={"kind": "jsonl", "path": "/etc/passwd"})
    with pytest.raises(ValidationError):
        make_spec(source={"kind": "dataset"})
    with pytest.raises(ValidationError):
        make_spec(index={"alias": "Bad Alias"})
    assert make_spec().build_collection("b1") == "filings_idx__b1"
    assert make_spec().embedding.space_id == f"tei-1.8-float16:{MODEL}@{'a' * 40}"


def test_load_spec_requires_matching_file_name(tmp_path: Path) -> None:
    path = tmp_path / "other.toml"
    path.write_text(
        'id = "filings"\ndescription = "x"\ntenant = "acme"\n'
        '[source]\nkind = "jsonl"\npath = "f.jsonl"\n'
        f'[embedding]\nmodel = "{MODEL}"\nrevision = "{"a" * 40}"\ndimensions = 32\n'
        '[index]\nalias = "filings_idx"\n'
    )
    with pytest.raises(ValueError, match="file name"):
        load_spec(path)


def test_gates_reject_bad_input_before_any_write() -> None:
    planned = plan(make_spec(), [document(1), document(1), document(2, tenant="other")])
    counts = planned.reports[0].counts
    assert counts["error:duplicate_document_id"] == 1 and counts["error:tenant_mismatch"] == 1
    assert not planned.passed


def test_access_policy_defaults_and_classification_gate() -> None:
    governed = plan(make_spec(access={"default_groups": ["analysts"]}), [document(1)])
    assert governed.documents[0].allowed_groups == ["analysts"]
    assert all(c.allowed_groups == ["analysts"] for c in governed.chunks["doc-1"])
    own = plan(make_spec(access={"default_groups": ["analysts"]}), [document(2, groups=["legal"])])
    assert own.documents[0].allowed_groups == ["legal"]  # never widened
    secret = plan(make_spec(access={"classification": "restricted"}), [document(3)])
    assert secret.reports[0].counts["error:missing_acl_for_classification"] == 1


async def test_build_records_status_and_lineage_in_the_index(client: AsyncQdrantClient) -> None:
    spec = make_spec()
    record = await run_build(spec, client, "b1", [document(1), document(2)])
    assert record["status"] == "verified" and record["documents"] == 2
    assert all(report["passed"] for report in record["reports"])
    stored = await IndexRegistry(client).record("filings_idx__b1")
    assert (
        stored is not None
        and stored["status"] == "verified"
        and stored["chunks"] == record["chunks"]
    )
    store = await make_store(spec, "filings_idx__b1", client)
    payloads = [payload for payload, _ in await store.sample_points(10)]
    assert {p["build_id"] for p in payloads} == {"b1"}
    assert all(len(p["document_version"]) == 16 for p in payloads)


async def test_rejected_input_creates_no_collection(client: AsyncQdrantClient) -> None:
    with pytest.raises(BuildRejected) as rejected:
        await run_build(make_spec(), client, "bad", [document(1, tenant="other")])
    assert not await client.collection_exists("filings_idx__bad")
    assert rejected.value.reports[0]["passed"] is False


async def test_promotion_is_atomic_and_rollback_restores_the_previous_build(
    client: AsyncQdrantClient,
) -> None:
    spec, registry = make_spec(), IndexRegistry(client)
    await run_build(spec, client, "20261101t000000_aaaaaa", [document(1)], promote=True)
    await run_build(
        spec, client, "20261102t000000_bbbbbb", [document(1), document(2)], promote=True
    )
    assert await registry.current("filings_idx") == "filings_idx__20261102t000000_bbbbbb"
    statuses = {b.build_id: b.status for b in await registry.builds("filings_idx")}
    assert statuses == {
        "20261101t000000_aaaaaa": "superseded",
        "20261102t000000_bbbbbb": "promoted",
    }
    serving = await make_store(spec, "filings_idx", client)  # the engine reads the alias
    assert await serving.count() == 2 * len(plan(spec, [document(1)]).chunks["doc-1"])
    assert await registry.rollback("filings_idx") == "filings_idx__20261101t000000_aaaaaa"
    assert await registry.current("filings_idx") == "filings_idx__20261101t000000_aaaaaa"


async def test_promotion_refuses_unverified_missing_or_unrecorded_collections(
    client: AsyncQdrantClient,
) -> None:
    registry = IndexRegistry(client)
    with pytest.raises(PromotionError, match="does not exist"):
        await registry.promote("filings_idx", "nope")
    store = await make_store(make_spec(), "filings_idx__half", client)
    await store.create_empty_collection()
    with pytest.raises(PromotionError, match="build record"):
        await registry.promote("filings_idx", "filings_idx__half")
    await store.set_build_record({"status": "building"})
    with pytest.raises(PromotionError, match="not verified"):
        await registry.promote("filings_idx", "filings_idx__half")
    legacy = await make_store(make_spec(), "legacy_index", client)
    await legacy.create_empty_collection()
    assert await registry.promote("filings_idx", "legacy_index", adopt=True) is None


def test_readback_detects_tampered_payloads_and_vectors() -> None:
    planned = plan(make_spec(), [document(1)])
    chunk = planned.chunks["doc-1"][0].model_copy(update={"build_id": "b1"})
    payload = chunk.model_dump(mode="json") | {"published_day": chunk.published_at.toordinal()}
    expected = {chunk.id: chunk}
    ok = validate_readback(
        [(payload, [1.0] * DIMS)],
        expected,
        dimensions=DIMS,
        build_id="b1",
        stored=1,
        counts_by_document={"doc-1": (1, 1)},
    )
    assert ok.passed
    widened = payload | {"allowed_groups": ["everyone"]}
    bad = validate_readback(
        [(widened, [float("nan")] * DIMS)],
        expected,
        dimensions=DIMS,
        build_id="b1",
        stored=2,
        counts_by_document={"doc-1": (0, 1)},
    )
    assert {key.split(":")[1] for key in bad.counts} >= {
        "payload_differs_from_source",
        "non_finite_vector",
        "point_count_mismatch",
        "document_chunk_count_mismatch",
    }


async def test_acl_update_and_delete_without_reembedding(client: AsyncQdrantClient) -> None:
    spec = make_spec()
    await run_build(spec, client, "b1", [document(1), document(2)])
    store = await make_store(spec, "filings_idx__b1", client)
    changed = await store.update_document_acl("acme", "doc-1", ["legal"])
    assert changed == await store.count_document("acme", "doc-1") > 0
    groups = {
        tuple(p["allowed_groups"])
        for p, _ in await store.sample_points(50)
        if p["document_id"] == "doc-1"
    }
    assert groups == {("legal",)}
    assert await store.update_document_acl("other", "doc-1", []) == 0  # tenant-scoped
    assert await store.delete_document("acme", "doc-2") > 0
    assert await store.count_document("acme", "doc-2") == 0


def test_cli_plan_reports_gates_without_network(tmp_path: Path, capsys: Any) -> None:
    (tmp_path / "filings.jsonl").write_text(
        "\n".join(document(n).model_dump_json() for n in (1, 2)) + "\n"
    )
    spec_file = tmp_path / "filings.toml"
    spec_file.write_text(
        'id = "filings"\ndescription = "x"\ntenant = "acme"\n'
        '[source]\nkind = "jsonl"\npath = "filings.jsonl"\n'
        f'[embedding]\nmodel = "{MODEL}"\nrevision = "{"a" * 40}"\ndimensions = 32\n'
        '[index]\nalias = "filings_idx"\n'
    )
    assert cli.main(["plan", str(spec_file), "--data-root", str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["documents"] == 2 and result["passed"] is True
