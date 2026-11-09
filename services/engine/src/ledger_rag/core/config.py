import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Request fields owned by the provider or by typed settings; extra_body may not override them.
LLM_RESERVED_BODY_KEYS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "stream",
        "stream_options",
        "n",
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "seed",
        "reasoning_effort",
    }
)


def validate_service_url(endpoint: str, *, https: bool = False) -> None:
    """Reject non-HTTP(S) URLs and URLs carrying credentials, queries or fragments."""
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
    if https and parsed.scheme != "https":
        raise ValueError("Production reranker endpoints must use HTTPS")


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
    # Chunk profile the Qdrant collection was built with; checked against its metadata.
    chunk_profile: Literal["v1", "v2", "v3"] = "v1"
    # Summary cache version (``<prompt>/<model>``) a v3 collection was built from.
    chunk_summary_version: str = Field(default="", max_length=128, pattern=r"^[a-zA-Z0-9._:/@+-]*$")
    embedding_batch_size: int = Field(default=32, ge=1, le=2048)
    qdrant_quantization: Literal["none", "scalar", "binary"] = "none"
    qdrant_oversampling: float = Field(default=2.0, ge=1.0, le=16.0)
    reranker_url: str = ""
    # Optional voice-profile endpoint for VllmRerankingRetriever; receives the same key.
    reranker_voice_url: str = ""
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
    # JSON merged into OpenAI-compatible chat requests for server-specific controls, e.g.
    # vLLM: {"chat_template_kwargs": {"enable_thinking": false}, "top_k": 20}.
    llm_extra_body: dict[str, Any] = Field(default_factory=dict)
    upstream_max_retries: int = Field(default=2, ge=0, le=5)
    upstream_retry_base_seconds: float = Field(default=0.2, gt=0, le=10)
    upstream_retry_max_delay_seconds: float = Field(default=5, gt=0, le=60)
    # Grounded answers: numeric provenance, optional model verifier and abstention policy.
    # Abstention thresholds are uncalibrated placeholders until fitted on FinQA dev data.
    verification_mode: Literal["off", "flag", "enforce"] = "enforce"
    verification_relative_tolerance: float = Field(default=0.0, ge=0, le=0.05)
    verification_max_rounding_ratio: float = Field(default=0.05, ge=0, le=0.5)
    groundedness_verifier: Literal["off", "lettucedetect"] = "off"
    groundedness_verifier_url: str = ""
    groundedness_verifier_api_key: SecretStr | None = None
    groundedness_verifier_timeout_seconds: float = Field(default=1.5, gt=0, le=10)
    groundedness_verifier_min_score: float = Field(default=0.5, ge=0, le=1)
    abstain_min_top_score: float | None = None
    abstain_min_citation_coverage: float = Field(default=0.0, ge=0, le=1)
    abstain_max_tool_errors: int | None = Field(default=None, ge=0)

    # Query planning (planning/*). planner_enabled=false restores single-query retrieval.
    planner_enabled: bool = True
    planner_rewrite_mode: Literal["llm", "deterministic"] = "llm"
    planner_rewrite_timeout_seconds: float = Field(default=1.5, gt=0, le=10)
    planner_rewrite_max_chars: int = Field(default=400, ge=50, le=1000)
    planner_decompose: bool = True
    planner_llm_decompose: bool = False
    planner_max_subqueries: int = Field(default=4, ge=1, le=4)
    planner_voice_max_subqueries: int = Field(default=2, ge=1, le=4)
    planner_max_concurrency: int = Field(default=4, ge=1, le=8)
    planner_rrf_k: int = Field(default=60, ge=1, le=1000)
    planner_gazetteer_path: Path | None = None
    planner_company_group_prefix: str = Field(default="company:", max_length=64)
    planner_hard_company_filter: bool = False
    decision_backend: Literal["heuristic", "laya"] = "heuristic"
    decision_url: str = ""
    decision_api_key: SecretStr | None = None
    decision_chat_timeout_seconds: float = Field(default=0.4, gt=0, le=0.4)
    decision_voice_timeout_seconds: float = Field(default=0.15, gt=0, le=0.15)
    decision_min_confidence: float = Field(default=0.6, ge=0, le=1)

    # External structured evidence (external/*): official SEC EDGAR and FRED/ALFRED APIs.
    # Off by default; never offered to voice or closed-corpus benchmarks.
    external_data_enabled: bool = False
    external_required_group: str = Field(default="", max_length=256)
    external_allowed_tenants: list[str] = Field(default_factory=list, max_length=100)
    external_max_calls_per_turn: int = Field(default=2, ge=1, le=5)
    external_max_facts_per_call: int = Field(default=8, ge=1, le=12)
    external_timeout_seconds: float = Field(default=8, gt=0, le=30)
    external_cache_ttl_seconds: float = Field(default=900, ge=0, le=86400)
    external_cache_max_bytes: int = Field(default=64_000_000, ge=0, le=1_000_000_000)
    external_max_response_bytes: int = Field(default=16_000_000, ge=65_536, le=64_000_000)
    external_breaker_failures: int = Field(default=5, ge=1, le=100)
    external_breaker_reset_seconds: float = Field(default=30, gt=0, le=3600)
    sec_enabled: bool = True
    # SEC fair access: declare a contact; at most 10 requests/second across all replicas.
    sec_user_agent: str = Field(default="Ledger RAG tusharmahalya@gmail.com", max_length=200)
    sec_max_requests_per_second: int = Field(default=5, ge=1, le=10)
    fred_enabled: bool = True
    fred_api_key: SecretStr | None = None
    fred_max_requests_per_minute: int = Field(default=60, ge=1, le=120)

    # MCP adapter (mcp_adapter/*): Streamable HTTP at /mcp, OAuth resource server only.
    mcp_enabled: bool = False
    mcp_resource_url: str = Field(default="", max_length=2048)
    mcp_audience: str = Field(default="", max_length=2048, pattern=r"^\S*$")
    mcp_required_scopes: list[str] = Field(default_factory=list, max_length=8)
    mcp_allowed_origins: list[str] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def validate_external(self) -> "Settings":
        if self.external_data_enabled and self.sec_enabled:
            agent = self.sec_user_agent.strip()
            # SEC requires a declared User-Agent with contact details; refuse to send without.
            if (
                agent != self.sec_user_agent
                or not re.fullmatch(r"[\x20-\x7e]+", agent)
                or not re.search(r"[^\s@]+@[^\s@]+\.[^\s@]+", agent)
            ):
                raise ValueError("LEDGER_SEC_USER_AGENT must declare a contact e-mail address")
        for tenant in self.external_allowed_tenants:
            if not tenant or len(tenant) > 200:
                raise ValueError("External data tenant allowlist entries must be 1-200 chars")
        return self

    @model_validator(mode="after")
    def validate_mcp(self) -> "Settings":
        for scope in self.mcp_required_scopes:
            if not re.fullmatch(r"[\x21\x23-\x5b\x5d-\x7e]{1,128}", scope):
                raise ValueError("MCP scopes must be RFC 6749 scope tokens")
        for origin in self.mcp_allowed_origins:
            parsed = urlsplit(origin)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.path:
                raise ValueError("MCP allowed origins must be bare HTTP(S) origins")
        if not self.mcp_enabled:
            return self
        parsed = urlsplit(self.mcp_resource_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path != "/mcp"
        ):
            raise ValueError("LEDGER_MCP_RESOURCE_URL must be an HTTP(S) URL whose path is /mcp")
        if self.environment == "production" and parsed.scheme != "https":
            raise ValueError("Production MCP resource URL must use HTTPS")
        if self.environment == "production" and self.auth_mode != "oidc":
            raise ValueError("Production MCP requires OIDC")
        return self

    @property
    def mcp_token_audience(self) -> str:
        """Audience MCP tokens must carry (RFC 8707); defaults to the resource URL."""
        return self.mcp_audience or self.mcp_resource_url

    @model_validator(mode="after")
    def validate_grounding(self) -> "Settings":
        if self.groundedness_verifier == "off":
            return self
        parsed = urlsplit(self.groundedness_verifier_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ) and not (self.environment == "development" and parsed.scheme == "http"):
            raise ValueError("Groundedness verifier requires an HTTPS URL without credentials")
        if not self.groundedness_verifier_api_key:
            raise ValueError("Groundedness verifier requires an API key")
        return self

    @field_validator("llm_extra_body")
    @classmethod
    def validate_llm_extra_body(cls, value: dict[str, Any]) -> dict[str, Any]:
        reserved = sorted(LLM_RESERVED_BODY_KEYS & set(value))
        if reserved:
            raise ValueError(
                f"LEDGER_LLM_EXTRA_BODY cannot set {', '.join(reserved)}; "
                "use the typed LEDGER_LLM_* settings"
            )
        if len(json.dumps(value)) > 4096:
            raise ValueError("LEDGER_LLM_EXTRA_BODY is limited to 4 KiB of JSON")
        return value

    @field_validator("planner_gazetteer_path", mode="before")
    @classmethod
    def empty_gazetteer_path(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def validate_planner(self) -> "Settings":
        if self.decision_backend == "laya":
            parsed = urlsplit(self.decision_url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("Laya decisions require an HTTP URL without credentials")
            if self.decision_api_key is None:
                raise ValueError("Laya decisions require LEDGER_DECISION_API_KEY")
            if self.environment == "production" and parsed.scheme != "https":
                raise ValueError("Production decision endpoint must use HTTPS")
        return self

    @model_validator(mode="after")
    def validate_deployment(self) -> "Settings":
        for endpoint in [
            self.llm_base_url,
            self.embedding_base_url,
            self.reranker_url,
            self.reranker_voice_url,
            self.qdrant_url,
            self.oidc_jwks_url,
            self.otlp_endpoint,
            self.stt_base_url,
            self.tts_base_url,
        ]:
            if endpoint:
                validate_service_url(endpoint)
        if self.chunk_profile == "v3" and not self.chunk_summary_version:
            raise ValueError("Chunk profile v3 requires LEDGER_CHUNK_SUMMARY_VERSION")
        if self.chunk_profile != "v3" and self.chunk_summary_version:
            raise ValueError("LEDGER_CHUNK_SUMMARY_VERSION applies only to chunk profile v3")
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
            for reranker in [self.reranker_url, self.reranker_voice_url]:
                if reranker and not reranker.startswith("https://"):
                    raise ValueError("Production reranker endpoints must use HTTPS")
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
    """Settings from ``LEDGER_*`` environment variables only (secrets included).

    Vault-rendered secret files were removed; a leftover ``LEDGER_SECRETS_DIR`` fails
    loudly instead of starting without the credentials it used to provide.
    """
    if os.environ.get("LEDGER_SECRETS_DIR"):
        raise ValueError(
            "LEDGER_SECRETS_DIR is no longer supported; deliver secrets as environment "
            "variables (docs/SECRETS.md)"
        )
    return Settings()
