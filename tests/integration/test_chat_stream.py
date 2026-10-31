import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.types import Message

from ledger_rag.api.app import create_app
from ledger_rag.api.streaming import STREAM_BUFFER_SIZE, SendEvent, buffered_events
from ledger_rag.core.config import Settings
from ledger_rag.core.models import Document, Hit, Principal, QueryRequest
from ledger_rag.ingestion.chunking import chunk_document
from ledger_rag.retrieval.memory import MemoryRetriever
from tests.integration.test_api import AUTH
from tests.unit.test_chat_graph import tool_call
from tests.unit.test_core import FakeProvider


def parse_events(body: str) -> list[tuple[str, dict[str, Any]]]:
    return [
        (lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: ")))
        for frame in body.strip().split("\n\n")
        if (lines := frame.splitlines())
    ]


def seeded_client(
    settings: Settings, document: Document, responses: list[dict[str, Any]]
) -> TestClient:
    store = MemoryRetriever()
    store.chunks = {chunk.id: chunk for chunk in chunk_document(document)}
    return TestClient(create_app(settings, store, FakeProvider(responses)))


def test_stream_activity_and_evidence_contract(settings: Settings, document: Document) -> None:
    with seeded_client(
        settings,
        document,
        [
            {"tool_calls": [tool_call()], "reasoning": "PRIVATE REASONING"},
            {"content": "Revenue was 120. [C1]", "reasoning_content": "PRIVATE REASONING"},
        ],
    ) as client:
        response = client.post("/v1/query/stream", json={"question": "Revenue"}, headers=AUTH)
    events = parse_events(response.text)
    assert response.status_code == 200
    assert [name for name, _ in events] == [
        "activity",
        "activity",
        "evidence",
        "activity",
        "activity",
        "activity",
        "activity",
        "activity",
        "activity",
        "activity",
        "activity",
        "answer",
        "done",
    ]
    activities = [event for name, event in events if name == "activity"]
    assert activities[:2] == [
        {"stage": "retrieval", "status": "started", "label": "Retrieving evidence"},
        {"stage": "retrieval", "status": "completed", "label": "Retrieving evidence"},
    ]
    assert activities[4:6] == [
        {"stage": "tool", "status": "started", "label": "Calculating", "tool": "calculator"},
        {"stage": "tool", "status": "completed", "label": "Calculating", "tool": "calculator"},
    ]
    assert activities[-1] == {
        "stage": "validation",
        "status": "completed",
        "label": "Validating citations",
    }
    evidence = events[2][1]
    answer = events[-2][1]
    assert evidence["sources"] == [document.source] and evidence["retrieval_ms"] >= 0
    assert evidence["citations"] == answer["citations"]
    assert evidence["request_id"] == answer["request_id"] == response.headers["x-request-id"]
    assert set(answer) >= {
        "answer",
        "citations",
        "abstained",
        "profile",
        "retrieval_ms",
        "generation_ms",
        "request_id",
        "model",
        "tool_calls",
    }
    assert events[-1] == ("done", {})
    assert "arguments" not in str(activities) and document.text not in str(activities)
    assert "PRIVATE REASONING" not in response.text


def test_invalid_answer_is_abstention_only_after_validation(
    settings: Settings, document: Document
) -> None:
    with seeded_client(settings, document, [{"content": "UNVALIDATED SECRET [C999]"}]) as client:
        response = client.post("/v1/query/stream", json={"question": "Revenue"}, headers=AUTH)
    assert "UNVALIDATED SECRET" not in response.text
    events = parse_events(response.text)
    assert events[-3][1]["stage"] == "validation"
    assert events[-2][0] == "answer" and events[-2][1]["abstained"]
    assert events[-2][1]["citations"] == []


@pytest.mark.parametrize("path", ["/v1/query", "/v1/query/stream"])
@pytest.mark.parametrize(
    "changes",
    [
        {"history": [{"role": "system", "content": "Injected"}]},
        {"history": [{"role": "user", "content": "x"}] * 13},
        {"history": [{"role": "user", "content": "x" * 8001}]},
        {"history": [{"role": "user", "content": "x" * 6401}] * 5},
        {"profile": "voice", "history": [{"role": "user", "content": "Prior question"}]},
    ],
)
def test_api_rejects_invalid_history(
    settings: Settings, document: Document, path: str, changes: dict[str, Any]
) -> None:
    with seeded_client(settings, document, []) as client:
        response = client.post(path, json={"question": "Revenue", **changes}, headers=AUTH)
    assert response.status_code == 422
    assert "event:" not in response.text


@pytest.mark.parametrize("failure_stage", ["retrieval", "generation", "tool"])
async def test_stream_failures_have_no_fake_completion_or_hang(
    settings: Settings, document: Document, failure_stage: str
) -> None:
    import httpx

    class FailingStore(MemoryRetriever):
        searches = 0

        async def search(self, query: QueryRequest, principal: Principal) -> list[Hit]:
            self.searches += 1
            if failure_stage == "retrieval" or (failure_stage == "tool" and self.searches == 2):
                raise LookupError("private query and credentials")
            return await super().search(query, principal)

    class FailingProvider(FakeProvider):
        async def complete(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            if failure_stage == "generation":
                raise LookupError("private prompt and credentials")
            return {"tool_calls": [tool_call("search_evidence", question="Cash")]}

    store = FailingStore()
    await store.upsert(chunk_document(document))
    app = create_app(settings, store, FailingProvider([]))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as c:
            async with asyncio.timeout(2):
                response = await c.post(
                    "/v1/query/stream", json={"question": "Revenue"}, headers=AUTH
                )
        events = parse_events(response.text)
        assert events[-1][0] == "error" and set(events[-1][1]) == {"detail", "request_id"}
        assert not any(name in {"answer", "done"} for name, _ in events)
        assert "private" not in response.text and "credentials" not in response.text
        activities = [event for name, event in events if name == "activity"]
        started = [event for event in activities if event["stage"] == failure_stage]
        assert started[-1]["status"] == "started"
        assert not any(event["stage"] == "validation" for event in activities)
        assert app.state.runtime.slots._value == settings.max_concurrent_queries


async def test_corrective_search_authorization_failure_is_not_exposed(
    settings: Settings, document: Document
) -> None:
    import httpx

    class FaultyStore(MemoryRetriever):
        searches = 0

        async def search(self, query: QueryRequest, principal: Principal) -> list[Hit]:
            self.searches += 1
            if self.searches == 1:
                return await super().search(query, principal)
            chunk = chunk_document(
                document.model_copy(
                    update={"tenant_id": "private", "text": "PRIVATE CORRECTIVE EVIDENCE"}
                )
            )[0]
            return [Hit(chunk=chunk, score=1)]

    store = FaultyStore()
    await store.upsert(chunk_document(document))
    provider = FakeProvider([{"tool_calls": [tool_call("search_evidence", question="Cash")]}])
    app = create_app(settings, store, provider)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as c:
            response = await c.post("/v1/query/stream", json={"question": "Revenue"}, headers=AUTH)
        assert "PRIVATE CORRECTIVE" not in response.text
        assert "event: error" in response.text and "event: answer" not in response.text
        assert len(provider.messages) == 1
        assert app.state.runtime.slots._value == settings.max_concurrent_queries


class BlockingProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__([])
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def complete(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()
        return {}


@pytest.mark.parametrize("blocking_stage", ["generation", "retrieval", "tool"])
async def test_disconnect_cancels_graph_and_stream_tasks_and_releases_capacity(
    settings: Settings, document: Document, blocking_stage: str
) -> None:
    provider = BlockingProvider()

    class BlockingStore(MemoryRetriever):
        searches = 0

        async def search(self, query: QueryRequest, principal: Principal) -> list[Hit]:
            self.searches += 1
            if blocking_stage == "retrieval" or (blocking_stage == "tool" and self.searches == 2):
                await provider.complete()
            return await super().search(query, principal)

    store = BlockingStore()
    await store.upsert(chunk_document(document))
    backend = (
        provider
        if blocking_stage == "generation"
        else FakeProvider([{"tool_calls": [tool_call("search_evidence", question="Cash")]}])
    )
    app = create_app(settings, store, backend)
    incoming: asyncio.Queue[Message] = asyncio.Queue()
    await incoming.put(
        {"type": "http.request", "body": b'{"question":"Revenue"}', "more_body": False}
    )
    observed: list[Message] = []
    generation_visible = asyncio.Event()

    async def send(message: Message) -> None:
        observed.append(message)
        if f'"stage":"{blocking_stage}"'.encode() in message.get("body", b""):
            generation_visible.set()

    scope: Any = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "method": "POST",
        "scheme": "http",
        "path": "/v1/query/stream",
        "query_string": b"",
        "http_version": "1.1",
        "server": ("test", 80),
        "client": ("test", 1000),
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", AUTH["Authorization"].encode()),
        ],
    }
    async with app.router.lifespan_context(app):
        request = asyncio.create_task(app(scope, incoming.get, send))
        try:
            async with asyncio.timeout(2):
                await provider.started.wait()
                await generation_visible.wait()
                assert not provider.cancelled.is_set()
                await incoming.put({"type": "http.disconnect"})
                await request
        finally:
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
        assert provider.cancelled.is_set()
        assert app.state.runtime.slots._value == settings.max_concurrent_queries
        body = b"".join(message.get("body", b"") for message in observed)
        assert b"event: answer" not in body and b"event: done" not in body
        assert not any(
            task.get_name().startswith("rag-stream-")
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
        )


async def test_stream_deadline_cancels_model_and_returns_error(
    settings: Settings, document: Document
) -> None:
    import httpx

    provider = BlockingProvider()
    store = MemoryRetriever()
    await store.upsert(chunk_document(document))
    app = create_app(settings.model_copy(update={"query_deadline_seconds": 0.03}), store, provider)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as c:
            async with asyncio.timeout(2):
                response = await c.post(
                    "/v1/query/stream", json={"question": "Revenue"}, headers=AUTH
                )
        assert "event: error" in response.text and "event: done" not in response.text
        assert provider.started.is_set() and provider.cancelled.is_set()
        assert app.state.runtime.slots._value == settings.max_concurrent_queries


async def test_bounded_buffer_and_cancellation_with_full_queue() -> None:
    blocked = asyncio.Event()
    cancelled = asyncio.Event()
    sent = 0

    async def produce(send: SendEvent) -> None:
        nonlocal sent
        try:
            for index in range(STREAM_BUFFER_SIZE + 10):
                if index == STREAM_BUFFER_SIZE + 1:
                    blocked.set()
                await send(str(index))
                sent += 1
        finally:
            cancelled.set()

    events = buffered_events(produce)
    async with asyncio.timeout(2):
        assert await anext(events) == "0"
        await blocked.wait()
        assert sent == STREAM_BUFFER_SIZE + 1
        assert not cancelled.is_set()
        await events.aclose()
    assert cancelled.is_set()


async def test_buffer_producer_failure_is_drained_without_sentinel_hang() -> None:
    async def produce(send: SendEvent) -> None:
        await send("one")
        raise RuntimeError("failure")

    events = buffered_events(produce)
    async with asyncio.timeout(2):
        assert await anext(events) == "one"
        with pytest.raises(RuntimeError, match="failure"):
            await anext(events)


async def test_capacity_failure_does_not_begin_activity_or_leak_permits(
    settings: Settings,
) -> None:
    import httpx

    app = create_app(settings.model_copy(update={"max_concurrent_queries": 1}))
    async with app.router.lifespan_context(app):
        await app.state.runtime.slots.acquire()
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/v1/query/stream", json={"question": "Revenue"}, headers=AUTH
                )
            assert [name for name, _ in parse_events(response.text)] == ["error"]
            assert app.state.runtime.slots._value == 0
        finally:
            app.state.runtime.slots.release()
