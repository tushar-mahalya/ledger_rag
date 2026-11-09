import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from ledger_voice.config import VoiceSettings
from ledger_voice.engine import (
    ABSTENTION_PROMPT,
    BUSY_PROMPT,
    FAILURE_PROMPT,
    TOO_LONG_PROMPT,
    EngineAnswer,
    EngineClient,
    is_meaningful,
    to_speakable,
)

ROOT = Path(__file__).resolve().parents[3]


def answer(text: str, *, abstained: bool = False, profile: str = "voice") -> EngineAnswer:
    return EngineAnswer(answer=text, abstained=abstained, profile=profile, request_id="r1")  # type: ignore[arg-type]


def test_speakable_strips_markers_and_never_cuts_a_sentence() -> None:
    text, outcome = to_speakable(answer("Revenue was **USD 120 million** [C1]."), 400)
    assert (text, outcome) == ("Revenue was USD 120 million.", "answered")
    two = "Revenue was 120 million in 2024 [C1]. It was 100 million in 2023 [C2]."
    assert to_speakable(answer(two), 40) == ("Revenue was 120 million in 2024.", "answered")
    assert to_speakable(answer("x" * 100), 40) == (TOO_LONG_PROMPT, "too_long")


def test_abstained_empty_or_wrong_profile_answers_are_never_spoken() -> None:
    assert to_speakable(answer("anything", abstained=True), 400)[0] == ABSTENTION_PROMPT
    assert to_speakable(answer("[C1]"), 400)[0] == ABSTENTION_PROMPT
    # A chat-profile answer skipped the voice bounds; do not read it out.
    assert to_speakable(answer("Long chat answer.", profile="chat"), 400)[0] == ABSTENTION_PROMPT


@pytest.mark.parametrize(
    ("text", "meaningful"),
    [("What was revenue in 2024?", True), ("um", False), ("Thank you.", False), ("a", False)],
)
def test_noise_transcripts_skip_the_engine(text: str, meaningful: bool) -> None:
    assert is_meaningful(text) is meaningful


def settings(**overrides: object) -> VoiceSettings:
    return VoiceSettings(engine_token=SecretStr("service-token"), **overrides)  # type: ignore[arg-type]


async def test_engine_client_contract_and_failure_mapping() -> None:
    seen: list[httpx.Request] = []
    responses = iter(
        [
            httpx.Response(
                200,
                json={
                    "answer": "It was 5 [C1].",
                    "abstained": False,
                    "profile": "voice",
                    "request_id": "q1",
                    "citations": [],
                },
            ),
            httpx.Response(429, json={"detail": "Rate limit exceeded"}),
            httpx.Response(503, text="upstream detail that must not be spoken"),
            httpx.Response(200, text="not json"),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return next(responses)

    client = EngineClient(settings(max_transcript_chars=20), httpx.MockTransport(handler))
    first = await client.answer("  What was the revenue figure for Aurora in fiscal 2024?  ")
    assert (first.text, first.outcome, first.request_id) == ("It was 5.", "answered", "q1")
    body = json.loads(seen[0].content)
    assert body == {"question": "What was the revenue", "profile": "voice"}
    assert seen[0].headers["Authorization"] == "Bearer service-token"
    assert seen[0].url.path == "/v1/query"
    assert (await client.answer("q")).text == BUSY_PROMPT
    assert (await client.answer("q")).text == FAILURE_PROMPT
    assert (await client.answer("q")).text == FAILURE_PROMPT
    await client.close()


async def test_engine_transport_errors_become_a_fixed_phrase() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused: secret-host")

    client = EngineClient(settings(), httpx.MockTransport(handler))
    reply = await client.answer("What was revenue?")
    assert (reply.text, reply.outcome) == (FAILURE_PROMPT, "failed")
    await client.close()


def test_generated_engine_fixtures_satisfy_the_voice_contract() -> None:
    fixtures = json.loads((ROOT / "contracts/answers.json").read_text())
    parsed = [
        EngineAnswer.model_validate(value)
        for key, value in fixtures.items()
        if isinstance(value, dict) and "answer" in value and "abstained" in value
    ]
    assert len(parsed) >= 4
    assert any(item.abstained for item in parsed) and any(not item.abstained for item in parsed)
