import os
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LEDGER_",
        env_file=None,
        secrets_dir=None,
        extra="ignore",
        hide_input_in_errors=True,
    )
    environment: Literal["development", "production"] = "development"
    auth_mode: Literal["demo", "oidc"] = "demo"
    demo_token: SecretStr | None = None
    oidc_issuer: str = ""
    oidc_audience: str = ""
    oidc_jwks_url: str = ""
    provider: Literal["extractive", "openai", "anthropic"] = "extractive"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: SecretStr | None = None
    llm_model: str = ""
    llm_timeout_seconds: float = Field(default=20, gt=0, le=120)
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: SecretStr | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, ge=32, le=8192)
    embedding_space_id: str = Field(default="", max_length=256, pattern=r"^[a-zA-Z0-9._:/@+-]*$")
    # Instruction prefix some embedding models expect on queries only (e.g. BGE, E5, Arctic).
    embedding_query_prefix: str = Field(default="", max_length=200)
    retrieval_fusion: Literal["hybrid", "dense", "sparse"] = "hybrid"
    embedding_batch_size: int = Field(default=32, ge=1, le=2048)
    qdrant_quantization: Literal["none", "scalar", "binary"] = "none"
    qdrant_oversampling: float = Field(default=2.0, ge=1.0, le=16.0)
    reranker_url: str = ""
    reranker_api_key: SecretStr | None = None
    reranker_timeout_seconds: float = Field(default=3, gt=0, le=60)
    rerank_candidates: int = Field(default=20, ge=1, le=100)
    retrieval_backend: Literal["memory", "qdrant"] = "memory"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "ledger_chunks"
    qdrant_upsert_batch_size: int = Field(default=128, ge=1, le=2048)
    seed_path: Path | None = None
    rate_limit_per_minute: int = Field(default=60, ge=1)
    max_concurrent_queries: int = Field(default=32, ge=1, le=1000)
    query_deadline_seconds: float = Field(default=25, gt=0, le=180)
    max_tool_rounds: int = Field(default=3, ge=1, le=5)
    otlp_endpoint: str = ""
    otlp_headers: dict[str, SecretStr] = Field(default_factory=dict)
    voice_enabled: bool = False
    stt_base_url: str = "https://api.openai.com/v1"
    stt_api_key: SecretStr | None = None
    stt_model: str = "whisper-1"
    stt_language: str = Field(default="en", pattern=r"^$|^[a-z]{2,3}$")
    stt_timeout_seconds: float = Field(default=10, gt=0, le=60)
    tts_base_url: str = "https://api.openai.com/v1"
    tts_api_key: SecretStr | None = None
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "alloy"
    tts_format: Literal["pcm", "wav"] = "pcm"
    tts_pcm_sample_rate: int = Field(default=24000, ge=8000, le=48000)
    tts_speed: float = Field(default=1.0, ge=0.25, le=4.0)
    tts_timeout_seconds: float = Field(default=10, gt=0, le=60)
    max_voice_sessions: int = Field(default=16, ge=1, le=1000)
    voice_max_session_seconds: float = Field(default=900, gt=0, le=7200)
    voice_max_frame_bytes: int = Field(default=32768, ge=1024, le=1_048_576)
    voice_max_response_seconds: float = Field(default=60, gt=0, le=600)
    voice_max_answer_chars: int = Field(default=400, ge=40, le=4000)
    vad_speech_rms: int = Field(default=500, ge=1, le=32767)
    vad_min_speech_ms: int = Field(default=200, ge=20, le=2000)
    vad_barge_in_ms: int = Field(default=300, ge=20, le=2000)
    vad_trailing_silence_ms: int = Field(default=500, ge=100, le=5000)
    vad_max_utterance_seconds: float = Field(default=15, ge=1, le=60)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # Reliability controls. Unset generation fields are omitted from provider requests.
    qdrant_bootstrap_on_startup: bool = False
    qdrant_write_lease_seconds: float = Field(default=120, ge=5, le=86400)
    llm_temperature: float | None = Field(default=None, ge=0, le=2)
    llm_max_tokens: int | None = Field(default=None, ge=1, le=131072)
    llm_max_tokens_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    llm_seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    llm_reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None

    @model_validator(mode="after")
    def validate_deployment(self) -> "Settings":
        for endpoint in [
            self.llm_base_url,
            self.embedding_base_url,
            self.reranker_url,
            self.qdrant_url,
            self.oidc_jwks_url,
            self.otlp_endpoint,
            self.stt_base_url,
            self.tts_base_url,
        ]:
            if not endpoint:
                continue
            parsed = urlsplit(endpoint)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("Service endpoints must be HTTP URLs without embedded credentials")
        if self.auth_mode == "demo" and (
            self.demo_token is None or len(self.demo_token.get_secret_value()) < 8
        ):
            raise ValueError("Demo auth requires LEDGER_DEMO_TOKEN of at least 8 characters")
        if self.auth_mode == "oidc" and not all(
            [self.oidc_issuer, self.oidc_audience, self.oidc_jwks_url]
        ):
            raise ValueError("OIDC issuer, audience and JWKS URL are required")
        if self.environment == "production":
            if self.auth_mode != "oidc" or self.retrieval_backend != "qdrant":
                raise ValueError("Production requires OIDC and persistent Qdrant retrieval")
            if not self.oidc_jwks_url.startswith("https://"):
                raise ValueError("Production JWKS URL must use HTTPS")
            if not self.oidc_issuer.startswith("https://"):
                raise ValueError("Production OIDC issuer must use HTTPS")
            if self.provider == "extractive":
                raise ValueError("Production requires an explicitly configured LLM")
            if not self.llm_base_url.startswith("https://"):
                raise ValueError("Production LLM endpoint must use HTTPS")
            if not self.embedding_base_url.startswith("https://"):
                raise ValueError("Production embedding endpoint must use HTTPS")
            if self.reranker_url and not self.reranker_url.startswith("https://"):
                raise ValueError("Production reranker endpoint must use HTTPS")
            if not self.stt_base_url.startswith("https://") or not self.tts_base_url.startswith(
                "https://"
            ):
                raise ValueError("Production speech endpoints must use HTTPS")
        if self.provider != "extractive" and (not self.llm_api_key or not self.llm_model):
            raise ValueError("Remote LLM requires model and API key")
        if self.retrieval_backend == "qdrant" and not self.embedding_api_key:
            raise ValueError("Qdrant retrieval requires an embedding API key")
        return self


@lru_cache
def get_settings() -> Settings:
    secrets_dir = os.environ.get("LEDGER_SECRETS_DIR")
    if secrets_dir and not Path(secrets_dir).is_dir():
        raise ValueError("LEDGER_SECRETS_DIR must refer to an existing directory")
    if not secrets_dir:
        return Settings()

    class VaultSettings(Settings):
        model_config = SettingsConfigDict(
            env_prefix="LEDGER_",
            env_file=None,
            secrets_dir=secrets_dir,
            extra="ignore",
            hide_input_in_errors=True,
        )

    return VaultSettings()
