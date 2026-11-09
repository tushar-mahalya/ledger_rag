import json
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError

from ledger_rag.core.models import Document
from ledger_rag.ingestion import context
from ledger_rag.ingestion.bulk import IngestSpec, ingest_documents
from ledger_rag.ingestion.context import (
    SummaryError,
    document_hash,
    load_summaries,
    summarize_documents,
    summary_cache_dir,
)
from ledger_rag.retrieval.qdrant import QdrantRetriever


def page(identifier: str, text: str = "Revenue was 123.") -> Document:
    return Document(
        id=identifier,
        tenant_id="finqa",
        title=f"{identifier} title",
        text=text,
        source="https://example.test/filing",
        published_at=date(2025, 1, 1),
    )


def completion(content: str | None) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


async def test_summaries_retry_rate_limits_cache_by_content_and_skip_cached(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    in_flight = peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        body = json.loads(request.content)
        title = body["messages"][1]["content"].split("\n")[0]
        calls.append(title)
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            if title == "a title" and calls.count(title) == 1:
                return httpx.Response(429)
            return completion(f"  Summary   of {title}. ")
        finally:
            in_flight -= 1

    cache = summary_cache_dir(tmp_path)
    docs = [page("a"), page("b"), page("c")]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://llm.test/v1/"
    ) as client:
        stats = await summarize_documents(docs, cache, client, concurrency=2, base_delay=0)
        assert stats == {"cached": 0, "generated": 3, "failed": 0}
        assert calls.count("a title") == 2
        assert peak <= 2
        again = await summarize_documents(docs, cache, client, base_delay=0)
    assert again == {"cached": 3, "generated": 0, "failed": 0}
    summaries = load_summaries(cache)
    assert summaries[document_hash("a title", "Revenue was 123.")] == "Summary of a title."
    assert cache.parts[-2:] == (context.SUMMARY_PROMPT_VERSION, context.SUMMARY_MODEL)


async def test_summaries_fail_fast_on_client_errors_and_bound_concurrency(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400) if b"bad" in request.content else completion("")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://llm.test/"
    ) as client:
        stats = await summarize_documents(
            [page("bad"), page("empty")], tmp_path, client, attempts=2, base_delay=0
        )
        assert stats == {
            "cached": 0,
            "generated": 0,
            "failed": 2,
            "failed: LLM returned HTTP 400": 1,
            "failed: Empty summary": 1,
        }
        with pytest.raises(ValueError, match="concurrency"):
            await summarize_documents([], tmp_path, client, concurrency=9)
    assert load_summaries(tmp_path) == {}


def test_corrupt_summary_cache_is_rejected(tmp_path: Path) -> None:
    entry = tmp_path / "ab" / "abc.json"
    entry.parent.mkdir()
    entry.write_text(json.dumps({"document_hash": "other", "summary": "x"}))
    with pytest.raises(SummaryError):
        load_summaries(tmp_path)
    with pytest.raises(ValueError):
        summary_cache_dir(tmp_path, "../escape")


async def test_bulk_ingestion_applies_profiles_and_requires_v3_summaries() -> None:
    store = AsyncMock(spec=QdrantRetriever)
    store.count.side_effect = [0, 1]
    result = await ingest_documents([page("a")], store, tenant_id="finqa", chunk_profile="v2")
    assert result.chunk_profile == "v2"
    [chunk] = store.append_staging.await_args.args[0]
    assert chunk.index_text.startswith("Document: a title\n\n")

    store = AsyncMock(spec=QdrantRetriever)
    store.count.side_effect = [0, 1]
    summaries = {document_hash("a title", "Revenue was 123."): "Revenue page."}
    await ingest_documents(
        [page("a")], store, tenant_id="finqa", chunk_profile="v3", summaries=summaries
    )
    [chunk] = store.append_staging.await_args.args[0]
    assert "\nSummary: Revenue page.\n\n" in chunk.index_text

    store = AsyncMock(spec=QdrantRetriever)
    store.count.return_value = 0
    with pytest.raises(ValueError, match="cached summary"):
        await ingest_documents([page("b")], store, tenant_id="finqa", chunk_profile="v3")


def test_ingest_spec_validates_chunk_profile() -> None:
    spec = IngestSpec(collection="c", model_revision="a" * 40, chunk_profile="v2")
    assert spec.chunk_profile == "v2"
    assert IngestSpec(collection="c", model_revision="a" * 40).chunk_profile == "v1"
    with pytest.raises(ValidationError):
        IngestSpec(collection="c", model_revision="a" * 40, chunk_profile="v4")  # type: ignore[arg-type]
