import asyncio
import json
from email.utils import formatdate
from time import time

import httpx
import pytest

from ledger_rag.core.config import Settings
from ledger_rag.core.models import Document, Principal, QueryRequest
from ledger_rag.core.retry import (
    RetryPolicy,
    request_deadline,
    retry_after_seconds,
    retry_budget,
    send_with_retries,
)
from ledger_rag.ingestion.chunking import chunk_document
from ledger_rag.retrieval.memory import MemoryRetriever
from ledger_rag.retrieval.reranking import RerankingRetriever

POLICY = RetryPolicy(max_retries=3, base_delay=0.1, max_delay=2.0)


def scripted(*outcomes: int | Exception) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []
    remaining = list(outcomes)

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        outcome = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        if isinstance(outcome, Exception):
            raise outcome
        return httpx.Response(outcome, json={"ok": outcome == 200})

    return httpx.AsyncClient(transport=httpx.MockTransport(handle)), requests


async def send(client: httpx.AsyncClient, policy: RetryPolicy = POLICY) -> httpx.Response:
    request = client.build_request("POST", "https://upstream.test/x", json={"q": 1})
    return await send_with_retries(lambda: client.send(request), policy, operation="test")


async def test_transient_statuses_retry_with_bounded_full_jitter(
    no_retry_sleep: list[float],
) -> None:
    client, requests = scripted(503, 429, 200)
    response = await send(client)
    assert response.status_code == 200 and len(requests) == 3
    assert all(json.loads(request.content) == {"q": 1} for request in requests)
    assert 0 <= no_retry_sleep[0] <= 0.1 and 0 <= no_retry_sleep[1] <= 0.2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422, 501])
async def test_non_transient_statuses_are_not_retried(status: int) -> None:
    client, requests = scripted(status)
    assert (await send(client)).status_code == status and len(requests) == 1


async def test_retries_stop_at_policy_limit_and_return_last_response() -> None:
    client, requests = scripted(502)
    assert (await send(client)).status_code == 502 and len(requests) == 4


async def test_transport_errors_retry_then_reraise() -> None:
    client, requests = scripted(httpx.ConnectError("refused"), 200)
    assert (await send(client)).status_code == 200 and len(requests) == 2
    client, requests = scripted(httpx.ConnectError("refused"))
    with pytest.raises(httpx.ConnectError):
        await send(client)
    assert len(requests) == 4


def test_retry_after_parses_seconds_and_http_dates() -> None:
    assert retry_after_seconds(httpx.Response(429, headers={"Retry-After": "1.5"})) == 1.5
    dated = httpx.Response(503, headers={"Retry-After": formatdate(time() + 30, usegmt=True)})
    assert 25 <= (retry_after_seconds(dated) or 0) <= 31
    assert retry_after_seconds(httpx.Response(503, headers={"Retry-After": "soon"})) is None
    assert retry_after_seconds(httpx.Response(503)) is None


async def test_retry_after_is_honored_or_abandons_when_too_long(
    no_retry_sleep: list[float],
) -> None:
    requests = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(429 if requests == 1 else 200, headers={"Retry-After": "1.5"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    assert (await send(client)).status_code == 200 and no_retry_sleep == [1.5]

    long_wait = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(503, headers={"Retry-After": "120"})
        )
    )
    assert (await send(long_wait)).status_code == 503 and no_retry_sleep == [1.5]


async def test_deadline_without_room_for_another_attempt_does_not_retry() -> None:
    client, requests = scripted(503, 200)
    with retry_budget(0.2):
        assert (await send(client)).status_code == 503
    assert len(requests) == 1
    client, requests = scripted(httpx.ReadTimeout("slow"), 200)
    with retry_budget(0.2), pytest.raises(httpx.ReadTimeout):
        await send(client)
    assert len(requests) == 1


async def test_retry_after_beyond_remaining_deadline_does_not_retry() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(429, headers={"Retry-After": "1.9"})
        )
    )
    with retry_budget(2.0):
        assert (await send(client)).status_code == 429


async def test_voice_deadline_allows_at_most_one_retry() -> None:
    client, requests = scripted(503)
    async with request_deadline(10, voice=True):
        assert (await send(client)).status_code == 503
    assert len(requests) == 2
    client, requests = scripted(503)
    async with request_deadline(10):
        await send(client)
    assert len(requests) == 4


async def test_request_deadline_still_times_out() -> None:
    with pytest.raises(TimeoutError):
        async with request_deadline(0.01):
            await asyncio.sleep(1)


async def test_reranker_retries_transient_failures(
    settings: Settings, document: Document, principal: Principal
) -> None:
    store = MemoryRetriever()
    await store.upsert(chunk_document(document))
    wrapper = RerankingRetriever(store, settings.model_copy(update={"reranker_url": "http://tei"}))
    await wrapper.client.aclose()
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=[{"index": 0, "score": 0.5}])

    wrapper.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    ranked = await wrapper.search(QueryRequest(question="Revenue"), principal)
    assert len(requests) == 2 and ranked[0].score == 0.5
    await wrapper.close()


def test_retry_settings_are_bounded(settings: Settings) -> None:
    assert RetryPolicy.from_settings(settings) == RetryPolicy(2, 0.2, 5.0)
    for update in [
        {"upstream_max_retries": 6},
        {"upstream_retry_base_seconds": 0},
        {"upstream_retry_max_delay_seconds": 61},
    ]:
        with pytest.raises(ValueError):
            Settings.model_validate({**settings.model_dump(), **update})
