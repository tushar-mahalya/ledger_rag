"""Voice service settings from ``LEDGER_VOICE_*`` environment variables (via ledgerctl)."""

import json
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

E164 = r"^\+[1-9][0-9]{6,14}$"


class VoiceSettings(BaseSettings):
    # Field names avoid the engine's own ``voice_*`` settings (LEDGER_VOICE_ENABLED,
    # LEDGER_VOICE_MAX_*), which share this prefix until ADR 0006 removes them.
    model_config = SettingsConfigDict(
        env_prefix="LEDGER_VOICE_", env_file=None, extra="ignore", hide_input_in_errors=True
    )

    environment: Literal["development", "production"] = "development"

    # Engine contract: POST {engine_url}/v1/query with profile=voice as a service principal.
    engine_url: str = "http://127.0.0.1:8080"
    engine_token: SecretStr | None = None
    engine_timeout_seconds: float = Field(default=8.0, gt=0, le=30)

    # ElevenLabs: Scribe v2 Realtime in, Flash v2.5 out; telephony audio stays at 8 kHz.
    elevenlabs_api_key: SecretStr | None = None
    stt_language: str = Field(default="en", pattern=r"^[a-z]{2,3}$")
    tts_voice_id: str = Field(default="", max_length=64, pattern=r"^[A-Za-z0-9]*$")
    tts_model: str = Field(default="eleven_flash_v2_5", max_length=64)
    # Optional JSON gazetteer ({"TICKER": ["alias", ...]}) biasing STT towards names.
    keyterms_path: Path | None = None
    max_keyterms: int = Field(default=50, ge=0, le=100)

    # Twilio: request signatures, auto hang-up and the lines this service answers.
    public_url: str = ""
    twilio_account_sid: str = Field(default="", pattern=r"^$|^AC[0-9a-f]{32}$")
    twilio_auth_token: SecretStr | None = None
    # Comma-separated or JSON list; NoDecode lets an empty value mean no lines.
    allowed_numbers: Annotated[list[str], NoDecode] = Field(default_factory=list, max_length=50)
    stream_secret: SecretStr | None = None
    stream_token_seconds: int = Field(default=60, ge=10, le=600)

    # Bounds.
    max_calls: int = Field(default=16, ge=1, le=1000)
    max_call_seconds: int = Field(default=900, ge=30, le=7200)
    max_transcript_chars: int = Field(default=600, ge=20, le=4000)
    max_spoken_chars: int = Field(default=400, ge=40, le=4000)
    max_message_bytes: int = Field(default=65_536, ge=1024, le=1_048_576)

    @field_validator("allowed_numbers", mode="before")
    @classmethod
    def parse_numbers(cls, value: object) -> object:
        if isinstance(value, str):
            value = json.loads(value) if value.strip().startswith("[") else value.split(",")
        return (
            [item.strip() for item in value if item.strip()] if isinstance(value, list) else value
        )

    @field_validator("allowed_numbers")
    @classmethod
    def validate_numbers(cls, value: list[str]) -> list[str]:
        import re

        for number in value:
            if not re.fullmatch(E164, number):
                raise ValueError("allowed numbers must be E.164, e.g. +15551234567")
        return value

    @field_validator("keyterms_path", mode="before")
    @classmethod
    def empty_path(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def validate_deployment(self) -> "VoiceSettings":
        for url in (self.engine_url, self.public_url):
            if not url:
                continue
            parts = urlsplit(url)
            if parts.scheme not in {"http", "https"} or not parts.hostname:
                raise ValueError("Voice URLs must be absolute HTTP(S) URLs")
            if parts.username or parts.password or parts.query or parts.fragment:
                raise ValueError("Voice URLs must not carry credentials, queries or fragments")
        if self.environment == "production":
            if not self.engine_url.startswith("https://"):
                raise ValueError("Production engine URL must use HTTPS")
            if not self.public_url.startswith("https://"):
                raise ValueError("Production public URL must use HTTPS")
            missing = [
                name
                for name, value in (
                    ("LEDGER_VOICE_TWILIO_AUTH_TOKEN", self.twilio_auth_token),
                    ("LEDGER_VOICE_STREAM_SECRET", self.stream_secret),
                    ("LEDGER_VOICE_ALLOWED_NUMBERS", self.allowed_numbers),
                )
                if not value
            ]
            if missing:
                raise ValueError(f"Production voice requires {', '.join(missing)}")
        if self.stream_secret is not None and len(self.stream_secret.get_secret_value()) < 32:
            raise ValueError("LEDGER_VOICE_STREAM_SECRET must be at least 32 characters")
        return self

    def require_live(self) -> None:
        """Fail loudly when a live call is attempted without the services it needs."""
        missing = [
            name
            for name, value in (
                ("LEDGER_VOICE_ENGINE_TOKEN", self.engine_token),
                ("LEDGER_VOICE_ELEVENLABS_API_KEY", self.elevenlabs_api_key),
                ("LEDGER_VOICE_TTS_VOICE_ID", self.tts_voice_id),
                ("LEDGER_VOICE_STREAM_SECRET", self.stream_secret),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(f"Voice calls need {', '.join(missing)} (ledgerctl set ...)")

    def keyterms(self) -> list[str]:
        """Company names and tickers from the gazetteer, bounded for the STT API."""
        if self.keyterms_path is None or self.max_keyterms == 0:
            return []
        data = json.loads(self.keyterms_path.read_text())
        if not isinstance(data, dict):
            raise ValueError("Keyterms gazetteer must be a JSON object")
        terms: list[str] = []
        for ticker, aliases in sorted(data.items()):
            for term in [ticker, *(aliases if isinstance(aliases, list) else [])]:
                if isinstance(term, str) and 1 < len(term) <= 50 and term not in terms:
                    terms.append(term)
        return terms[: self.max_keyterms]
