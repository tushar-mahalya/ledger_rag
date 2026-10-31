import asyncio
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import SecretStr, ValidationError
from qdrant_client import AsyncQdrantClient

from ledger_rag.core.config import Settings
from ledger_rag.core.models import Document
from ledger_rag.ingestion.bulk import IngestSpec, ingest_documents, jsonl_documents
from ledger_rag.ingestion.chunking import chunk_document
from ledger_rag.retrieval.qdrant import QdrantRetriever, RetrievalError


def document(identifier: str = "doc", text: str = "Revenue was 123.") -> Document:
    return Document(
        id=identifier,
        tenant_id="finqa",
        title="Annual report",
        text=text,
        source="https://example.test/filing",
        published_at=date(2025, 1, 1),
        allowed_groups=["finance"],
    )


def fake_store() -> AsyncMock:
    store = AsyncMock(spec=QdrantRetriever)
    store.count.side_effect = [0, 3]
    return store


async def test_batches_preserve_complete_documents_and_exact_count() -> None:
    docs = [document(str(i)) for i in range(3)]
    store = fake_store()
    result = await ingest_documents(iter(docs), store, tenant_id="finqa", batch_size=2)
    assert (result.documents, result.chunks, result.batches) == (3, 3, 2)
    assert len(result.corpus_sha256) == 64
    assert result.seconds >= 0
    assert [
        [c.document_id for c in call.args[0]] for call in store.append_staging.await_args_list
    ] == [
        ["0", "1"],
        ["2"],
    ]


async def test_large_document_is_not_split_across_authoritative_upserts() -> None:
    doc = document(text="x" * 8000)
    chunks = chunk_document(doc)
    assert len(chunks) > 2
    store = fake_store()
    store.count.side_effect = [0, len(chunks)]
    result = await ingest_documents([doc], store, tenant_id="finqa", batch_size=2)
    assert result.chunks == len(chunks)
    store.append_staging.assert_awaited_once()
    assert store.append_staging.call_args.args[0] == chunks


@pytest.mark.parametrize(
    ("docs", "error"),
    [
        ([], "no documents"),
        ([document(), document()], "Duplicate document"),
        ([document().model_copy(update={"tenant_id": "other"})], "tenant"),
    ],
)
async def test_invalid_input_fails_without_upload(docs: list[Document], error: str) -> None:
    store = fake_store()
    with pytest.raises(ValueError, match=error):
        await ingest_documents(docs, store, tenant_id="finqa")
    store.append_staging.assert_not_called()


async def test_nonempty_and_count_mismatch_are_not_success() -> None:
    store = fake_store()
    store.count.side_effect = [1]
    with pytest.raises(RetrievalError, match="empty"):
        await ingest_documents([document()], store, tenant_id="finqa")
    store.append_staging.assert_not_called()
    store.count.side_effect = [0, 0]
    with pytest.raises(RetrievalError, match="count mismatch"):
        await ingest_documents([document()], store, tenant_id="finqa")


async def test_upload_failure_stops_further_batches_and_verification() -> None:
    store = fake_store()
    store.append_staging.side_effect = RetrievalError("uncertain write")
    with pytest.raises(RetrievalError, match="uncertain"):
        await ingest_documents(
            [document("1"), document("2")], store, tenant_id="finqa", batch_size=1
        )
    store.append_staging.assert_awaited_once()
    store.count.assert_awaited_once()


def test_jsonl_validation_does_not_disclose_document(tmp_path: Path) -> None:
    source = tmp_path / "input.jsonl"
    source.write_text(document().model_dump_json() + "\n")
    assert list(jsonl_documents(source)) == [document()]
    source.write_text('{"text":"private-financial-data"}\n')
    with pytest.raises(ValueError, match="line 1") as error:
        list(jsonl_documents(source))
    assert "private-financial-data" not in str(error.value)


@pytest.mark.parametrize(
    "updates",
    [
        {"model_revision": "main"},
        {"collection": "bad/name"},
        {"model_id": "/etc/passwd"},
        {"batch_size": 0},
        {"batch_size": 513},
        {"tenant_id": "other"},
        {"source": "jsonl", "input_path": "/etc/passwd"},
        {"source": "jsonl", "input_path": "/data/../secret"},
        {"source": "jsonl", "input_path": ""},
    ],
)
def test_job_spec_rejects_mutable_revision_and_unsafe_input(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        IngestSpec.model_validate(
            {"collection": "new_index", "model_revision": "a" * 40, **updates}
        )


def test_embedding_space_pins_model_revision_and_runtime() -> None:
    spec = IngestSpec(collection="new_index", model_revision="a" * 40)
    assert spec.embedding_space_id == (
        "tei-1.8-float16:Alibaba-NLP/gte-modernbert-base@" + "a" * 40
    )
    assert (
        spec.embedding_space_id
        != spec.model_copy(update={"model_revision": "b" * 40}).embedding_space_id
    )


@pytest.mark.parametrize("fail_upload", [False, True])
async def test_modal_job_writes_database_directly_and_commits_only_verified_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fail_upload: bool
) -> None:
    import json

    pytest.importorskip("modal", reason="Modal orchestration tests require the compute group")
    from compute import ingestion
    from ledger_rag.evaluation.benchmarks import finqa
    from ledger_rag.ingestion import tei
    from ledger_rag.observability import telemetry
    from ledger_rag.retrieval import qdrant

    monkeypatch.setenv("LEDGER_QDRANT_URL", "https://database.example.test")
    monkeypatch.setenv("LEDGER_QDRANT_API_KEY", "database-private-key")
    monkeypatch.setattr(ingestion, "DATA", tmp_path)
    commit = AsyncMock()
    monkeypatch.setattr(
        ingestion,
        "data_volume",
        SimpleNamespace(
            reload=SimpleNamespace(aio=AsyncMock()), commit=SimpleNamespace(aio=commit)
        ),
    )
    monkeypatch.setattr(telemetry, "configure_telemetry", lambda settings: None)
    monkeypatch.setattr(
        ingestion.asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=SimpleNamespace(wait=AsyncMock(return_value=0))),
    )
    monkeypatch.setattr(
        finqa, "load", lambda path: (finqa.Corpus(documents={"doc": document()}), [])
    )
    settings = Settings(
        demo_token=SecretStr("offline-test"),
        embedding_api_key=SecretStr("loopback-test"),
        embedding_dimensions=32,
    )
    store = QdrantRetriever(settings)
    await store._client.close()
    store._client = AsyncQdrantClient(":memory:")
    await store._embedding_client.aclose()
    requests: list[httpx.Request] = []
    events: list[str] = []

    def embed(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert events == ["model_started"]
        if fail_upload:
            return httpx.Response(503, text="private-provider-body")
        inputs = json.loads(request.content)["input"]
        return httpx.Response(
            200, json={"data": [{"index": i, "embedding": [1.0] * 32} for i in range(len(inputs))]}
        )

    store._embedding_client = httpx.AsyncClient(transport=httpx.MockTransport(embed))
    original_close = store.close
    store.close = AsyncMock()

    def make_store(config: Settings) -> QdrantRetriever:
        store._settings = config
        store._collection = config.qdrant_collection
        return store

    monkeypatch.setattr(qdrant, "QdrantRetriever", make_store)

    @asynccontextmanager
    async def server(spec: IngestSpec, api_key: str):
        assert spec.model_revision == "a" * 40
        assert api_key != "database-private-key"
        assert store._settings.embedding_base_url == "http://127.0.0.1:8000/v1"
        assert store._settings.embedding_space_id == spec.embedding_space_id
        events.append("model_started")
        try:
            yield
        finally:
            events.append("model_stopped")

    monkeypatch.setattr(tei, "embedding_server", server)
    spec = IngestSpec(collection="new_index", model_revision="a" * 40, dimensions=32)
    try:
        if fail_upload:
            with pytest.raises(RuntimeError, match="inspect the isolated collection") as error:
                await ingestion.ingest.local(spec.model_dump(), "git-test")
            assert "private-provider-body" not in str(error.value)
            commit.assert_not_called()
            assert not await asyncio.to_thread(lambda: list(tmp_path.rglob("*.json")))
        else:
            report = await ingestion.ingest.local(spec.model_dump(), "git-test")
            assert report["status"] == "verified"
            assert report["chunks"] == 1
            assert report["documents"] == 1
            assert report["git_sha"] == "git-test"
            assert "database-private-key" not in json.dumps(report)
            text = await asyncio.to_thread(Path(report["report_path"]).read_text)
            assert json.loads(text) == report
            assert (await store._client.count("new_index", exact=True)).count == 1
            commit.assert_awaited_once()
        assert events == ["model_started", "model_stopped"]
        assert str(requests[0].url) == "http://127.0.0.1:8000/v1/embeddings"
        store.close.assert_awaited_once()
    finally:
        await original_close()
