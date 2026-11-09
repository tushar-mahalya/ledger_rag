import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.websockets import WebSocketDisconnect
from twilio.request_validator import RequestValidator

from ledger_voice.app import StreamStart, VoiceRuntime, create_app
from ledger_voice.config import VoiceSettings
from ledger_voice.tokens import TokenVerifier, mint

AUTH = "twilio-auth-token-for-tests"
SECRET = "stream-secret-" + "x" * 32
PUBLIC = "https://voice.example.test"
LINE = "+15550001111"


def settings(**overrides: Any) -> VoiceSettings:
    values: dict[str, Any] = {
        "engine_token": SecretStr("t"),
        "twilio_auth_token": SecretStr(AUTH),
        "twilio_account_sid": "AC" + "0" * 32,
        "stream_secret": SecretStr(SECRET),
        "public_url": PUBLIC,
        "allowed_numbers": [LINE],
    }
    values.update(overrides)
    return VoiceSettings(**values)


class Sessions:
    def __init__(self) -> None:
        self.started: list[StreamStart] = []

    async def __call__(self, websocket: Any, start: StreamStart, runtime: VoiceRuntime) -> None:
        self.started.append(start)


def signed(params: dict[str, str]) -> dict[str, str]:
    signature = RequestValidator(AUTH).compute_signature(f"{PUBLIC}/twilio/voice", params)
    return {"X-Twilio-Signature": signature}


def call(**extra: str) -> dict[str, str]:
    return {"CallSid": "CA" + "1" * 32, "To": LINE, "From": "+15559998888", **extra}


def test_webhook_requires_a_valid_twilio_signature() -> None:
    with TestClient(create_app(settings(), run_session=Sessions())) as client:
        assert client.post("/twilio/voice", data=call()).status_code == 403
        forged = {"X-Twilio-Signature": "AAAA"}
        assert client.post("/twilio/voice", data=call(), headers=forged).status_code == 403
        tampered = signed(call())
        response = client.post("/twilio/voice", data=call(To="+15550002222"), headers=tampered)
        assert response.status_code == 403


def test_webhook_connects_allowed_lines_with_a_call_bound_token() -> None:
    with TestClient(create_app(settings(), run_session=Sessions())) as client:
        params = call()
        response = client.post("/twilio/voice", data=params, headers=signed(params))
        assert response.status_code == 200
        body = response.text
        assert '<Stream url="wss://voice.example.test/twilio/media">' in body
        token = body.split('name="token" value="')[1].split('"')[0]
        assert TokenVerifier(SECRET, 60).verify(token, params["CallSid"])


def test_webhook_rejects_lines_this_service_does_not_answer() -> None:
    with TestClient(create_app(settings(), run_session=Sessions())) as client:
        params = call(To="+15550009999")
        response = client.post("/twilio/voice", data=params, headers=signed(params))
        assert response.status_code == 200 and "<Reject" in response.text


def test_webhook_is_unavailable_without_twilio_configuration() -> None:
    app = create_app(settings(twilio_auth_token=None), run_session=Sessions())
    with TestClient(app) as client:
        assert client.post("/twilio/voice", data=call()).status_code == 503


def start_message(token: str, call_sid: str = "CA" + "2" * 32) -> str:
    return json.dumps(
        {
            "event": "start",
            "start": {
                "streamSid": "MZ" + "3" * 32,
                "callSid": call_sid,
                "accountSid": "AC" + "0" * 32,
                "customParameters": {"token": token},
            },
        }
    )


def test_media_stream_starts_only_with_a_valid_single_use_token() -> None:
    sessions = Sessions()
    call_sid = "CA" + "2" * 32
    with TestClient(create_app(settings(), run_session=sessions)) as client:
        token = mint(SECRET, call_sid, 60)
        with client.websocket_connect("/twilio/media") as ws:
            ws.send_text(json.dumps({"event": "connected"}))
            ws.send_text(start_message(token, call_sid))
            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()
        assert [s.call_sid for s in sessions.started] == [call_sid]
        # Replaying the same token for the same call is refused.
        with client.websocket_connect("/twilio/media") as ws:
            ws.send_text(start_message(token, call_sid))
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_text()
        assert closed.value.code == 1008 and len(sessions.started) == 1


@pytest.mark.parametrize(
    "token",
    [mint(SECRET, "CA" + "9" * 32, 60), mint("other-secret" * 4, "CA" + "2" * 32, 60), "", "1.x"],
)
def test_media_stream_rejects_bad_tokens_before_any_media(token: str) -> None:
    sessions = Sessions()
    with TestClient(create_app(settings(), run_session=sessions)) as client:
        with client.websocket_connect("/twilio/media") as ws:
            ws.send_text(start_message(token))
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_text()
        assert closed.value.code == 1008 and sessions.started == []


def test_browser_origin_websockets_are_refused() -> None:
    sessions = Sessions()
    with TestClient(create_app(settings(), run_session=sessions)) as client:
        with pytest.raises(WebSocketDisconnect) as closed:
            with client.websocket_connect(
                "/twilio/media", headers={"Origin": "https://evil.example"}
            ) as ws:
                ws.receive_text()
        assert closed.value.code == 1008 and sessions.started == []


def test_tokens_expire_and_bind_to_the_call() -> None:
    verifier = TokenVerifier(SECRET, 60)
    token = mint(SECRET, "CA1", 60, now=1000)
    assert not verifier.verify(token, "CA1", now=1061)  # expired
    assert not verifier.verify(token, "CA2", now=1001)  # other call
    assert verifier.verify(token, "CA1", now=1001)
    assert not verifier.verify(mint(SECRET, "CA3", 600, now=1000), "CA3", now=1000)  # ttl too long


def test_production_requires_https_and_twilio_material() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        VoiceSettings(environment="production", engine_url="http://engine", public_url=PUBLIC)
    with pytest.raises(ValueError, match="LEDGER_VOICE_TWILIO_AUTH_TOKEN"):
        VoiceSettings(environment="production", engine_url="https://engine.test", public_url=PUBLIC)
    with pytest.raises(ValueError, match="E.164"):
        settings(allowed_numbers=["5550001111"])


def test_voice_variables_do_not_collide_with_engine_settings() -> None:
    from ledger_rag.core.config import Settings

    engine_fields = set(Settings.model_fields)
    for name in VoiceSettings.model_fields:
        assert f"voice_{name}" not in engine_fields, name


@pytest.mark.parametrize(
    ("raw", "parsed"),
    [
        ("", []),
        ("+15550001111", [LINE]),
        ("+15550001111, +15550002222", [LINE, "+15550002222"]),
        ('["+15550001111"]', [LINE]),
    ],
)
def test_allowed_numbers_parse_from_the_environment(
    raw: str, parsed: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEDGER_VOICE_ALLOWED_NUMBERS", raw)
    assert VoiceSettings().allowed_numbers == parsed
