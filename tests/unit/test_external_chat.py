import json
from pathlib import Path

import pytest

from ledger_rag.core.config import Settings
from ledger_rag.core.models import Activity, Principal, Profile, QueryRequest
from ledger_rag.generation.chat_tools import execute_tool
from ledger_rag.orchestration.service import RagService
from ledger_rag.retrieval.memory import MemoryRetriever
from tests.unit.external_fakes import FakeUpstream, external_data
from tests.unit.test_core import FakeProvider

READER = Principal(subject="analyst", tenant_id="demo", roles=["reader"], groups=[])


def test_recorded_fixtures_never_contain_credentials() -> None:
    root = Path(__file__).parents[1] / "fixtures" / "external"
    for path in root.rglob("*"):
        if path.is_file():
            contents = path.read_text().lower()
            assert "api_key" not in contents, path.name
            assert "0123456789abcdef0123456789abcdef" not in contents, path.name


async def test_chat_external_budget_activity_and_voice_isolation(settings: Settings) -> None:
    class Provider(FakeProvider):
        def __init__(self) -> None:
            self.offered: list[object] = []

        async def complete(self, messages: list, tools: list | None = None) -> dict:
            self.offered.append(tools)
            if len(self.offered) == 1:
                return {
                    "tool_calls": [
                        {
                            "id": f"call-{i}",
                            "name": "lookup_company_facts",
                            "arguments": {
                                "company": "AAPL",
                                "concepts": ["InterestExpense"],
                                "periods": ["FY2023"],
                            },
                        }
                        for i in range(3)
                    ]
                }
            return {"content": "Apple interest expense was $3,933 million [C1]."}

    upstream = FakeUpstream()
    data = external_data(upstream)
    provider = Provider()
    service = RagService(settings, MemoryRetriever(), provider)
    service.external = data
    events: list[Activity] = []

    async def record(event: Activity) -> None:
        events.append(event)

    answer = await service.generate(
        QueryRequest(question="Apple interest expense", as_of=data.as_of(None)),
        [],
        0,
        "test",
        READER,
        on_activity=record,
    )
    assert answer.citations and answer.citations[0].source_kind == "sec_xbrl"
    assert answer.tool_calls == 3
    assert [event.outcome for event in events if event.outcome] == ["ok", "ok", "error"]
    assert {tool["function"]["name"] for tool in provider.offered[0]} >= {
        "lookup_company_facts",
        "lookup_macro_series",
    }
    before = len(upstream.requests)
    await service.generate(
        QueryRequest(question="Apple interest expense", profile=Profile.VOICE),
        [],
        0,
        "voice",
        READER,
    )
    assert len(upstream.requests) == before
    await service.close()


async def test_unentitled_external_tools_not_offered(settings: Settings) -> None:
    from ledger_rag.external.service import ExternalSession
    from ledger_rag.external.tools import tool_schemas

    data = external_data(external_required_group="finance")
    session = ExternalSession(data, READER, data.as_of(None), data.providers(READER))
    assert tool_schemas(session) == []
    events: list[Activity] = []

    async def record(event: Activity) -> None:
        events.append(event)

    async def retrieve(query: QueryRequest, principal: Principal) -> tuple[list, float]:
        pytest.fail("No corpus fallback permitted")

    result = await execute_tool(
        {"id": "x", "name": "lookup_company_facts", "arguments": {"company": "AAPL"}},
        query=QueryRequest(question="lookup"),
        principal=READER,
        retrieve=retrieve,
        citations=[],
        evidence_ids=set(),
        on_activity=record,
        external=session,
    )
    assert "error" in json.loads(result["content"])
    assert events[-1].outcome == "unavailable"
    await data.aclose()
