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
    # Instruction prefix some embedding models expect on queries only (e.g. BGE, E5, Arctic).
    embedding_query_prefix: str = Field(default="", max_length=200)
    retrieval_fusion: Literal["hybrid", "dense", "sparse"] = "hybrid"
    reranker_url: str = ""
    reranker_api_key: SecretStr | None = None
    reranker_timeout_seconds: float = Field(default=3, gt=0, le=60)
    rerank_candidates: int = Field(default=20, ge=1, le=100)
    retrieval_backend: Literal["memory", "qdrant"] = "memory"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "ledger_chunks"
    seed_path: Path | None = None
    rate_limit_per_minute: int = Field(default=60, ge=1)
    max_concurrent_queries: int = Field(default=32, ge=1, le=1000)
    query_deadline_seconds: float = Field(default=25, gt=0, le=180)
    max_tool_rounds: int = Field(default=3, ge=1, le=5)
    otlp_endpoint: str = ""
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @model_validator(mode="after")
    def validate_deployment(self) -> "Settings":
        for endpoint in [
            self.llm_base_url,
            self.embedding_base_url,
            self.reranker_url,
            self.qdrant_url,
            self.oidc_jwks_url,
            self.otlp_endpoint,
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
