"""Regenerate the web client's answer-contract fixtures from the Python models.

Each scenario runs the real ``RagService`` chat pipeline offline: two synthetic Aurora
Devices disclosures in the in-memory retriever, the deterministic planner, a scripted
provider (one calculator call, then the final text) and a stub groundedness verifier.
Only timings are replaced, so the fixtures are byte-for-byte reproducible.

Writes ``web/src/fixtures/answers.json`` (answers, activity streams and evidence events,
serialized like the SSE route) and ``web/src/fixtures/contract.schema.json`` (Pydantic
JSON Schema for the zod bound-parity test in ``web/src/contract.test.ts``).

    uv run --frozen python scripts/dump_contract_fixtures.py          # rewrite
    uv run --frozen python scripts/dump_contract_fixtures.py --check  # fail when stale
"""

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

from pydantic.json_schema import models_json_schema

from ledger_rag.core.config import Settings
from ledger_rag.core.models import (
    Activity,
    Answer,
    Document,
    Evidence,
    Principal,
    QueryRequest,
)
from ledger_rag.generation.evidence import evidence_event
from ledger_rag.generation.verification import GroundednessResult
from ledger_rag.ingestion.chunking import chunk_document
from ledger_rag.orchestration.service import RagService
from ledger_rag.planning.analysis import Gazetteer
from ledger_rag.planning.planner import Planner
from ledger_rag.retrieval.memory import MemoryRetriever

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "web" / "src" / "fixtures"
ANSWERS = FIXTURES / "answers.json"
SCHEMA = FIXTURES / "contract.schema.json"

QUESTION = "What was the percentage change in Aurora Devices revenue from 2024 to 2025?"
FORECAST = "Forecast the share price for Aurora in 2027."
SUPPORTED = "Revenue rose from USD 120 million to USD 150 million, a 25% increase."
UNSUPPORTED = "Revenue rose to USD 150 million, a 31% increase over USD 115 million."
# Measured timings vary per run; fixtures pin them.
TIMINGS = {"retrieval_ms": 12.345, "generation_ms": 842.1}
REFUSAL_TIMINGS = {"retrieval_ms": 0.412, "generation_ms": 1.25}
LEGACY_FIELDS = {
    "answer",
    "citations",
    "abstained",
    "profile",
    "retrieval_ms",
    "generation_ms",
    "request_id",
    "model",
}

DOCUMENTS = [
    Document(
        id="aurora-2025",
        tenant_id="demo",
        title="Aurora Devices · FY2025",
        text=(
            "In fiscal year 2025, Aurora Devices reported revenue of USD 150 million, "
            "compared with USD 120 million in fiscal year 2024."
        ),
        source="urn:ledger:synthetic:aurora:2025",
        published_at=date(2026, 2, 1),
    ),
    Document(
        id="aurora-2024",
        tenant_id="demo",
        title="Aurora Devices · FY2024",
        text="In fiscal year 2024, Aurora Devices reported revenue of USD 120 million.",
        source="urn:ledger:synthetic:aurora:2024",
        published_at=date(2025, 2, 1),
    ),
]


class ScriptedProvider:
    """Requests the growth calculation once, then returns the final answer text."""

    model = "ledger-chat"

    def __init__(self, text: str) -> None:
        self.text = text
        # Citation id per document id, set from the retrieved evidence before generation.
        self.ids: dict[str, str] = {}
        self.calls = 0

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        self.calls += 1
        current, prior = self.ids["aurora-2025"], self.ids["aurora-2024"]
        if self.calls > 1:
            return {"content": f"{self.text} [{current}] [{prior}]"}
        later = {"citation_id": current, "quote": "USD 150 million"}
        earlier = {"citation_id": prior, "quote": "USD 120 million"}
        steps = [
            {"op": "subtract", "args": [later, earlier]},
            {"op": "divide", "args": [{"ref": "#0"}, earlier]},
            {"op": "multiply", "args": [{"ref": "#1"}, {"constant": 100}]},
        ]
        call = {"id": "call-1", "name": "calculator", "arguments": {"steps": steps}}
        return {"tool_calls": [call]}

    async def close(self) -> None:
        pass


class FixedGroundedness:
    name = "lettucedetect"

    async def verify(
        self, *, question: str, contexts: list[str], answer: str
    ) -> GroundednessResult:
        return GroundednessResult(score=0.93, unsupported_spans=[])

    async def close(self) -> None:
        pass


async def scenario(
    question: str, text: str, mode: str, request_id: str
) -> tuple[Answer, list[Activity], Evidence]:
    settings = Settings.model_construct(
        planner_rewrite_mode="deterministic", verification_mode=mode
    )
    gazetteer = Gazetteer()
    gazetteer.add("AURORA", "Aurora Devices", "Aurora")
    retriever = MemoryRetriever()
    for document in DOCUMENTS:
        await retriever.upsert(chunk_document(document))
    provider = ScriptedProvider(text)
    service = RagService(
        settings,
        retriever,
        provider,
        groundedness=FixedGroundedness(),
        planner=Planner(settings, gazetteer=gazetteer),
    )
    principal = Principal(subject="contract-fixtures", tenant_id="demo", roles=["reader"])
    query = QueryRequest(question=question)
    events: list[Activity] = []

    async def collect(event: Activity) -> None:
        events.append(event)

    try:
        hits, retrieval_ms = await service.retrieve(query, principal, on_activity=collect)
        evidence = evidence_event(request_id, retrieval_ms, hits)
        provider.ids = {item.document_id: item.id for item in evidence.citations}
        answer = await service.generate(
            query, hits, retrieval_ms, request_id, principal, on_activity=collect
        )
    finally:
        await service.close()
    return answer, events, evidence


async def build() -> dict[str, Any]:
    runs = {
        "verified": await scenario(QUESTION, SUPPORTED, "enforce", "req-verified"),
        "unverified": await scenario(QUESTION, UNSUPPORTED, "flag", "req-unverified"),
        "abstained": await scenario(QUESTION, UNSUPPORTED, "enforce", "req-abstained"),
        "refused": await scenario(FORECAST, SUPPORTED, "enforce", "req-refused"),
    }
    fixtures: dict[str, Any] = {
        "_generated_by": (
            "scripts/dump_contract_fixtures.py: the RagService chat pipeline run offline "
            "(memory retriever, deterministic planner, scripted provider). Do not edit by "
            "hand. Answers and evidence use model_dump(mode='json'), activities "
            "exclude_none=True like the SSE route, and timings are pinned."
        )
    }
    activities: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    for name, (answer, events, event) in runs.items():
        timings = REFUSAL_TIMINGS if name == "refused" else TIMINGS
        pinned = answer.model_copy(update=timings)
        fixtures[name] = Answer.model_validate(pinned.model_dump()).model_dump(mode="json")
        activities[name] = [item.model_dump(mode="json", exclude_none=True) for item in events]
        evidence[name] = event.model_copy(
            update={"retrieval_ms": timings["retrieval_ms"]}
        ).model_dump(mode="json")
    fixtures["legacy"] = {
        key: value for key, value in fixtures["verified"].items() if key in LEGACY_FIELDS
    }
    fixtures["legacy"]["request_id"] = "req-legacy"
    fixtures["activities"] = activities
    fixtures["evidence"] = evidence
    return fixtures


def schema() -> dict[str, Any]:
    _, definitions = models_json_schema(
        [(Answer, "serialization"), (Activity, "serialization"), (Evidence, "serialization")],
        ref_template="#/$defs/{model}",
    )
    return definitions


def render(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def outputs() -> dict[Path, str]:
    return {ANSWERS: render(asyncio.run(build())), SCHEMA: render(schema())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate the web contract fixtures.")
    parser.add_argument("--check", action="store_true", help="fail if the fixtures are stale")
    args = parser.parse_args()
    stale = {path: text for path, text in outputs().items() if _read(path) != text}
    if args.check:
        for path in stale:
            print(f"stale: {path.relative_to(ROOT)}", file=sys.stderr)
        if stale:
            print("run: uv run --frozen python scripts/dump_contract_fixtures.py", file=sys.stderr)
        return 1 if stale else 0
    for path, text in stale.items():
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


def _read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


if __name__ == "__main__":
    raise SystemExit(main())
