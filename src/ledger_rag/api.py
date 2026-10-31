import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

import httpx
import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ledger_rag.auth import Authenticator, RateLimiter
from ledger_rag.chunking import chunk_document
from ledger_rag.config import Settings, get_settings
from ledger_rag.contracts import Provider, Retriever
from ledger_rag.models import Answer, Document, IngestRequest, Principal, QueryRequest
from ledger_rag.retrieval import make_retriever
from ledger_rag.service import RagService
from ledger_rag.telemetry import BodyTooLarge, Metrics, RequestMiddleware, configure_telemetry

logger = structlog.get_logger()


@dataclass
class Runtime:
    settings: Settings
    service: RagService
    auth: Authenticator
    limiter: RateLimiter
    slots: asyncio.Semaphore
    metrics: Metrics


def create_app(
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    provider: Provider | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved = settings or get_settings()
        telemetry = configure_telemetry(resolved)
        store = retriever
        if store is None:
            store = make_retriever(resolved)
        backend = provider
        if backend is None:
            from ledger_rag.providers import make_provider

            backend = make_provider(resolved)
        try:
            if resolved.seed_path:
                if resolved.environment == "production":
                    raise ValueError(
                        "Seed data is disabled in production; use authenticated ingestion"
                    )
                documents = [
                    Document.model_validate_json(line)
                    for line in resolved.seed_path.read_text().splitlines()
                    if line.strip()
                ]
                await store.upsert([c for doc in documents for c in chunk_document(doc)])
            app.state.runtime = Runtime(
                settings=resolved,
                service=RagService(resolved, store, backend),
                auth=Authenticator(resolved),
                limiter=RateLimiter(resolved.rate_limit_per_minute),
                slots=asyncio.Semaphore(resolved.max_concurrent_queries),
                metrics=Metrics(),
            )
            logger.info(
                "service_started",
                environment=resolved.environment,
                retrieval=resolved.retrieval_backend,
                provider=resolved.provider,
            )
            yield
        finally:
            await store.close()
            await backend.close()
            if telemetry:
                await asyncio.to_thread(telemetry.shutdown)

    app = FastAPI(title="Ledger RAG", version="0.1.0", lifespan=lifespan)
    app.add_middleware(RequestMiddleware)

    def runtime(request: Request) -> Runtime:
        state: Runtime = request.app.state.runtime
        return state

    async def identity(
        request: Request, authorization: Annotated[str | None, Header()] = None
    ) -> Principal:
        state = runtime(request)
        principal = await state.auth.authenticate(authorization)
        state.limiter.check(principal)
        return principal

    Identity = Annotated[Principal, Depends(identity)]

    def require_role(principal: Principal, role: str) -> None:
        if role not in principal.roles:
            raise HTTPException(403, f"{role} role required")

    def observe(state: Runtime, answer: Answer) -> None:
        profile = answer.profile.value
        state.metrics.retrieval.labels(profile).observe(answer.retrieval_ms / 1000)
        state.metrics.generation.labels(profile).observe(answer.generation_ms / 1000)
        if answer.abstained:
            state.metrics.abstentions.labels(profile).inc()

    @asynccontextmanager
    async def slot(state: Runtime) -> AsyncIterator[None]:
        try:
            await asyncio.wait_for(state.slots.acquire(), timeout=0.01)
        except TimeoutError as exc:
            raise HTTPException(
                503, "Service is at capacity", headers={"Retry-After": "1"}
            ) from exc
        try:
            yield
        finally:
            state.slots.release()

    @app.exception_handler(BodyTooLarge)
    async def too_large(request: Request, exc: BodyTooLarge) -> JSONResponse:
        logger.warning("request_body_too_large")
        return JSONResponse(status_code=413, content={"detail": "Request body exceeds 2 MB"})

    @app.exception_handler(httpx.HTTPError)
    async def upstream_error(request: Request, exc: httpx.HTTPError) -> JSONResponse:
        logger.error("upstream_failure", error_class=type(exc).__name__)
        return JSONResponse(status_code=502, content={"detail": "Upstream service failed"})

    @app.exception_handler(ValueError)
    async def invalid_upstream(request: Request, exc: ValueError) -> JSONResponse:
        logger.error("invalid_upstream_response", error_class=type(exc).__name__)
        return JSONResponse(status_code=502, content={"detail": "Invalid upstream response"})

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error("internal_failure", error_class=type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "Internal service failure"})

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready(request: Request) -> dict[str, str]:
        state = runtime(request)
        try:
            async with asyncio.timeout(3):
                healthy = await state.service.retriever.ready()
        except (httpx.HTTPError, TimeoutError) as exc:
            logger.warning("readiness_dependency_failure", error_class=type(exc).__name__)
            raise HTTPException(503, "Retrieval dependency unavailable") from exc
        if not healthy:
            raise HTTPException(503, "Retrieval dependency unavailable")
        return {"status": "ready"}

    @app.get("/metrics", include_in_schema=False)
    async def metrics(request: Request, principal: Identity) -> Response:
        require_role(principal, "admin")
        return Response(
            generate_latest(runtime(request).metrics.registry), media_type=CONTENT_TYPE_LATEST
        )

    @app.post("/v1/documents", status_code=201)
    async def ingest(
        payload: IngestRequest, request: Request, principal: Identity
    ) -> dict[str, int]:
        require_role(principal, "admin")
        if any(doc.tenant_id != principal.tenant_id for doc in payload.documents):
            raise HTTPException(403, "Cross-tenant ingestion is forbidden")
        keys = [(doc.tenant_id, doc.id) for doc in payload.documents]
        if len(keys) != len(set(keys)):
            raise HTTPException(422, "Duplicate document identifiers in batch")
        chunks = [chunk for document in payload.documents for chunk in chunk_document(document)]
        state = runtime(request)
        async with slot(state):
            try:
                async with asyncio.timeout(state.settings.query_deadline_seconds):
                    await state.service.retriever.upsert(chunks)
            except TimeoutError as exc:
                raise HTTPException(504, "Ingestion deadline exceeded") from exc
        logger.info(
            "documents_ingested", document_count=len(payload.documents), chunk_count=len(chunks)
        )
        return {"documents": len(payload.documents), "chunks": len(chunks)}

    @app.post("/v1/query", response_model=Answer)
    async def query(payload: QueryRequest, request: Request, principal: Identity) -> Answer:
        require_role(principal, "reader")
        state = runtime(request)
        async with slot(state):
            try:
                async with asyncio.timeout(state.settings.query_deadline_seconds):
                    hits, retrieval_ms = await state.service.retrieve(payload, principal)
                    answer = await state.service.generate(
                        payload, hits, retrieval_ms, request.state.request_id, principal
                    )
            except TimeoutError as exc:
                logger.warning("query_deadline_exceeded")
                raise HTTPException(504, "Query deadline exceeded") from exc
        observe(state, answer)
        return answer

    @app.post("/v1/query/stream")
    async def stream(
        payload: QueryRequest, request: Request, principal: Identity
    ) -> StreamingResponse:
        require_role(principal, "reader")
        state = runtime(request)

        async def events() -> AsyncIterator[str]:
            # Emit evidence immediately; never send unvalidated model tokens to a voice client.
            try:
                async with slot(state):
                    async with asyncio.timeout(state.settings.query_deadline_seconds):
                        hits, retrieval_ms = await state.service.retrieve(payload, principal)
                        evidence_event = json.dumps(
                            {
                                "request_id": request.state.request_id,
                                "retrieval_ms": retrieval_ms,
                                "sources": [hit.chunk.source for hit in hits],
                            }
                        )
                        yield f"event: evidence\ndata: {evidence_event}\n\n"
                        answer = await state.service.generate(
                            payload, hits, retrieval_ms, request.state.request_id, principal
                        )
                        observe(state, answer)
                        yield f"event: answer\ndata: {answer.model_dump_json()}\n\n"
                        yield "event: done\ndata: {}\n\n"
            except (TimeoutError, httpx.HTTPError, ValueError, RuntimeError, HTTPException) as exc:
                logger.error("stream_failure", error_class=type(exc).__name__)
                error_event = json.dumps(
                    {
                        "detail": "Stream failed; do not treat partial events as a final answer",
                        "request_id": request.state.request_id,
                    }
                )
                yield f"event: error\ndata: {error_event}\n\n"

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"}
        )

    return app


app = create_app()
