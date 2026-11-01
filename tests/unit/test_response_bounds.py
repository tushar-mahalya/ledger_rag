from datetime import date

import pytest
from pydantic import ValidationError

from ledger_rag.core.models import (
    MAX_CITATIONS,
    MAX_QUOTE_CHARS,
    MAX_RESPONSE_JSON_BYTES,
    Answer,
    Calculation,
    CalculationOperand,
    CalculationStep,
    Citation,
    Evidence,
    Profile,
)


def citations() -> list[Citation]:
    return [
        Citation(
            id=f"C{index + 1}",
            document_id="d",
            source="urn:test",
            title="Synthetic",
            published_at=date(2025, 1, 1),
            quote="",
        )
        for index in range(MAX_CITATIONS)
    ]


@pytest.mark.parametrize("model", [Answer, Evidence])
@pytest.mark.parametrize("character", ["x", '"', "\u0001", "\u20ac"])
def test_response_json_budget_counts_utf8_escaping_and_metadata(
    model: type[Answer] | type[Evidence], character: str
) -> None:
    pool = citations()
    if model is Answer:
        response: Answer | Evidence = Answer(
            answer="Supported. [C1]",
            citations=pool,
            abstained=False,
            profile=Profile.CHAT,
            retrieval_ms=1,
            generation_ms=1,
            request_id="r",
            model="synthetic",
        )
    else:
        response = Evidence(request_id="r", retrieval_ms=1, sources=[], citations=pool)
    room = MAX_RESPONSE_JSON_BYTES - len(response.model_dump_json().encode("utf-8"))
    cost = len(
        Citation.model_validate(pool[0])
        .model_copy(update={"quote": character})
        .model_dump_json()
        .encode("utf-8")
    ) - len(pool[0].model_dump_json().encode("utf-8"))
    for item in pool[:-1]:
        length = min(MAX_QUOTE_CHARS, room // cost)
        item.quote = character * length
        room -= cost * length
    pool[-1].quote = "x" * room
    payload = response.model_dump(mode="json")
    accepted = model.model_validate(payload)
    assert len(accepted.model_dump_json().encode("utf-8")) == MAX_RESPONSE_JSON_BYTES
    payload["citations"][-1]["quote"] += "x"
    with pytest.raises(ValidationError, match="UTF-8 bytes"):
        model.model_validate(payload)


def test_field_valid_large_evidence_is_rejected_before_streaming() -> None:
    pool = citations()
    for item in pool:
        item.quote = "x" * MAX_QUOTE_CHARS
    with pytest.raises(ValidationError, match="UTF-8 bytes"):
        Evidence(request_id="r", retrieval_ms=1, sources=[], citations=pool)


def test_response_budget_includes_all_calculation_traces() -> None:
    operand = CalculationOperand(kind="evidence", value="1", citation_id="C1", quote="x" * 10_000)
    calculation = Calculation(
        steps=[
            CalculationStep(index=index, op="table_sum", operands=[operand] * 64, result="64")
            for index in range(16)
        ],
        result="64",
        citation_ids=["C1"],
    )
    with pytest.raises(ValidationError, match="UTF-8 bytes"):
        Answer(
            answer="Supported. [C1]",
            citations=citations()[:1],
            abstained=False,
            profile=Profile.CHAT,
            retrieval_ms=1,
            generation_ms=1,
            request_id="r",
            model="synthetic",
            calculation=calculation,
            calculations=[calculation],
        )
