import logging
import time
import uuid
from typing import Any

import structlog
from fastapi import HTTPException
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import CollectorRegistry, Counter, Histogram
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ledger_rag.core.config import Settings


class BodyTooLarge(HTTPException):
    def __init__(self) -> None:
        super().__init__(413, "Request body exceeds 2 MB")


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "ledger_http_requests_total",
            "Completed HTTP requests",
            ["route", "method", "status"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "ledger_http_duration_seconds",
            "End-to-end HTTP duration",
            ["route", "method"],
            registry=self.registry,
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
        )
        self.retrieval = Histogram(
            "ledger_retrieval_seconds", "Retrieval latency", ["profile"], registry=self.registry
        )
        self.generation = Histogram(
            "ledger_generation_seconds", "Generation latency", ["profile"], registry=self.registry
        )
        self.abstentions = Counter(
            "ledger_abstentions_total",
            "Answers without sufficient evidence",
            ["profile"],
            registry=self.registry,
        )


def configure_telemetry(settings: Settings) -> TracerProvider | None:
    logging.basicConfig(level=settings.log_level, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, settings.log_level)),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    if not settings.otlp_endpoint:
        return None
    provider = TracerProvider(
        resource=Resource.create({"service.name": "ledger-rag", "service.version": "0.1.0"})
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=f"{settings.otlp_endpoint.rstrip('/')}/v1/traces")
        )
    )
    trace.set_tracer_provider(provider)
    return provider


class RequestMiddleware:
    def __init__(self, app: ASGIApp, body_limit: int = 2_000_000) -> None:
        self.app = app
        self.body_limit = body_limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        size = 0
        status = 500
        start = time.perf_counter()

        async def limited_receive() -> Message:
            nonlocal size
            message = await receive()
            if message["type"] == "http.request":
                size += len(message.get("body", b""))
                if size > self.body_limit:
                    raise BodyTooLarge()
            return message

        async def wrapped_send(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = [
                    *message.get("headers", []),
                    (b"x-request-id", request_id.encode()),
                    (b"x-content-type-options", b"nosniff"),
                    (b"cache-control", b"no-store"),
                ]
            await send(message)

        tracer = trace.get_tracer(__name__)
        try:
            with tracer.start_as_current_span("http.request") as span:
                context = span.get_span_context()
                structlog.contextvars.bind_contextvars(trace_id=f"{context.trace_id:032x}")
                await self.app(scope, limited_receive, wrapped_send)
                span.set_attribute("http.response.status_code", status)
        finally:
            runtime: Any = getattr(scope["app"].state, "runtime", None)
            route = getattr(scope.get("route"), "path", "unmatched")
            duration = time.perf_counter() - start
            if runtime is not None:
                runtime.metrics.requests.labels(route, scope["method"], str(status)).inc()
                runtime.metrics.duration.labels(route, scope["method"]).observe(duration)
            structlog.get_logger().info(
                "http_request",
                route=route,
                method=scope["method"],
                status=status,
                duration_ms=round(duration * 1000, 3),
            )
            structlog.contextvars.clear_contextvars()
