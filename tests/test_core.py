import json
from datetime import date
from typing import Any

import pytest

from ledger_rag.chunking import chunk_document
from ledger_rag.config import Settings
from ledger_rag.models import Document, Principal, Profile, QueryRequest
from ledger_rag.retrieval import MemoryRetriever
from ledger_rag.service import RagService
from ledger_rag.tools import calculate


class FakeProvider:
    model = "test"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.messages: list[list[dict[str, Any]]] = []

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        self.messages.append(list(messages))
        return self.responses.pop(0)

    async def close(self) -> None:
        pass


def test_structural_chunking_is_stable_and_bounded(document: Document) -> None:
    doc = document.model_copy(
        update={"text": ("Revenue and cash data. " * 100) + "\n\nTable\n|A|B|"}
    )
    chunks = chunk_document(doc, max_chars=300, overlap=30)
    assert chunks == chunk_document(doc, max_chars=300, overlap=30)
    assert all(len(c.text) <= 300 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)
    assert any("|A|B|" in c.text for c in chunks)
    assert chunks[0].parent_text


@pytest.mark.parametrize("max_chars,overlap", [(100, 0), (200, 200), (300, -1)])
def test_invalid_chunking(document: Document, max_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_document(document, max_chars=max_chars, overlap=overlap)


async def test_retrieval_filters_before_ranking(document: Document, principal: Principal) -> None:
    store = MemoryRetriever()
    other = document.model_copy(update={"tenant_id": "other", "id": "other"})
    restricted = document.model_copy(update={"id": "restricted", "allowed_groups": ["secret"]})
    future = document.model_copy(update={"id": "future", "published_at": date(2026, 1, 1)})
    await store.upsert(
        [c for d in [document, other, restricted, future] for c in chunk_document(d)]
    )
    query = QueryRequest(question="Revenue", as_of=date(2025, 12, 31))
    hits = await store.search(query, principal)
    assert [h.chunk.document_id for h in hits] == ["doc-1"]
    privileged = principal.model_copy(update={"groups": ["secret"]})
    assert {h.chunk.document_id for h in await store.search(query, privileged)} == {
        "doc-1",
        "restricted",
    }
    assert not await store.search(QueryRequest(question="unfindablexyz"), principal)


async def test_reingestion_removes_stale_chunks(document: Document) -> None:
    store = MemoryRetriever()
    await store.upsert(chunk_document(document))
    old_ids = set(store.chunks)
    await store.upsert(
        chunk_document(document.model_copy(update={"text": "Entirely changed text."}))
    )
    assert not old_ids & set(store.chunks)
    assert len(store.chunks) == 1


@pytest.mark.parametrize(
    "expression,expected", [("0.1 + 0.2", "0.3"), ("(150-120)/120*100", "25.00"), ("-2*3", "-6")]
)
def test_calculator(expression: str, expected: str) -> None:
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "2**100",
        "1/0",
        "True + 1",
        "1e999",
        "[1]",
        "1 << 2",
        "9" * 201,
        "1e-1000000",
        "1e-128 * 0.1",
        "1e-1000000000",
    ],
)
def test_calculator_rejects_unsafe_inputs(expression: str) -> None:
    with pytest.raises(ValueError):
        calculate(expression)


def test_calculator_fixed_point_output_is_bounded() -> None:
    assert len(calculate("1e-128")) <= 256


async def test_citations_are_from_authorized_evidence(
    settings: Settings, document: Document, principal: Principal
) -> None:
    store = MemoryRetriever()
    await store.upsert(chunk_document(document))
    provider = FakeProvider([{"content": "Revenue was USD 120 million. [C1]", "tool_calls": []}])
    service = RagService(settings, store, provider)
    query = QueryRequest(question="Revenue")
    hits, elapsed = await service.retrieve(query, principal)
    answer = await service.generate(query, hits, elapsed, "test")
    assert not answer.abstained
    assert answer.citations[0].quote == document.text
    assert answer.citations[0].document_id == document.id


@pytest.mark.parametrize(
    "content", ["Guess with no citation", "False fact [C999]", "Fact [C1]\n\nUncited paragraph"]
)
async def test_invalid_citations_abstain(
    settings: Settings, document: Document, principal: Principal, content: str
) -> None:
    store = MemoryRetriever()
    await store.upsert(chunk_document(document))
    service = RagService(settings, store, FakeProvider([{"content": content}]))
    query = QueryRequest(question="Revenue")
    hits, elapsed = await service.retrieve(query, principal)
    answer = await service.generate(query, hits, elapsed, "test")
    assert answer.abstained and not answer.citations


async def test_tool_loop_is_bounded_and_decimal(
    settings: Settings, document: Document, principal: Principal
) -> None:
    store = MemoryRetriever()
    await store.upsert(chunk_document(document))
    provider = FakeProvider(
        [
            {
                "content": "",
                "tool_calls": [
                    {"id": "t1", "name": "calculator", "arguments": {"expression": "120*0.15"}}
                ],
            },
            {"content": "Computed income is USD 18 million. [C1]", "tool_calls": []},
        ]
    )
    service = RagService(settings, store, provider)
    query = QueryRequest(question="Revenue")
    hits, elapsed = await service.retrieve(query, principal)
    answer = await service.generate(query, hits, elapsed, "test")
    assert answer.tool_calls == 1
    tool_message = provider.messages[1][-1]
    assert json.loads(tool_message["content"])["result"] == "18.00"


async def test_voice_disables_tools(
    settings: Settings, document: Document, principal: Principal
) -> None:
    store = MemoryRetriever()
    await store.upsert(chunk_document(document))
    service = RagService(
        settings,
        store,
        FakeProvider(
            [
                {
                    "tool_calls": [
                        {"id": "t1", "name": "calculator", "arguments": {"expression": "1+1"}}
                    ]
                }
            ]
        ),
    )
    query = QueryRequest(question="Revenue", profile=Profile.VOICE)
    hits, elapsed = await service.retrieve(query, principal)
    answer = await service.generate(query, hits, elapsed, "test")
    assert answer.abstained and answer.tool_calls == 0
