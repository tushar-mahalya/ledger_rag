from collections.abc import Mapping
from typing import Any

from pydantic import SecretStr

from ledger_rag.core.config import Settings
from ledger_rag.observability import telemetry


def test_optional_otlp_endpoint_and_secret_headers(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class FakeExporter:
        def __init__(self, *, endpoint: str, headers: Mapping[str, str] | None) -> None:
            captured["endpoint"] = endpoint
            captured["headers"] = headers

    class FakeProcessor:
        def __init__(self, exporter: FakeExporter) -> None:
            captured["exporter"] = exporter

    class FakeProvider:
        def __init__(self, *, resource: Any) -> None:
            captured["resource"] = resource

        def add_span_processor(self, processor: FakeProcessor) -> None:
            captured["processor"] = processor

    monkeypatch.setattr(telemetry, "OTLPSpanExporter", FakeExporter)
    monkeypatch.setattr(telemetry, "BatchSpanProcessor", FakeProcessor)
    monkeypatch.setattr(telemetry, "TracerProvider", FakeProvider)
    monkeypatch.setattr(telemetry.trace, "set_tracer_provider", lambda provider: None)

    settings = Settings(
        demo_token=SecretStr("test-only-token"),
        otlp_endpoint="https://cloud.langfuse.com/api/public/otel/",
        otlp_headers={
            "Authorization": SecretStr("Basic redacted-test-credential"),
            "x-langfuse-ingestion-version": SecretStr("4"),
        },
    )
    provider = telemetry.configure_telemetry(settings)

    assert isinstance(provider, FakeProvider)
    assert captured["endpoint"] == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    assert captured["headers"] == {
        "Authorization": "Basic redacted-test-credential",
        "x-langfuse-ingestion-version": "4",
    }


def test_otlp_is_disabled_without_endpoint(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        telemetry, "OTLPSpanExporter", lambda **kwargs: (_ for _ in ()).throw(AssertionError())
    )
    settings = Settings(demo_token=SecretStr("test-only-token"))
    assert telemetry.configure_telemetry(settings) is None


def test_otlp_headers_load_from_environment(monkeypatch: Any) -> None:
    monkeypatch.setenv(
        "LEDGER_OTLP_HEADERS",
        '{"Authorization":"Basic redacted-test-credential","x-langfuse-ingestion-version":"4"}',
    )
    settings = Settings(
        demo_token=SecretStr("test-only-token"),
        otlp_endpoint="https://cloud.langfuse.com/api/public/otel",
    )
    assert settings.otlp_headers["Authorization"].get_secret_value() == (
        "Basic redacted-test-credential"
    )
    assert str(settings.otlp_headers["Authorization"]) == "**********"
