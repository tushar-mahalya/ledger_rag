import asyncio
import json
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import urlsplit

from fastapi import HTTPException
from mcp.server import Server, ServerRequestContext
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
    ToolAnnotations,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import Message, Receive, Scope, Send

from ledger_rag.core.config import Settings
from ledger_rag.core.models import MAX_RESPONSE_JSON_BYTES, Citation, Principal, QueryRequest
from ledger_rag.core.retry import request_deadline
from ledger_rag.external.errors import ExternalDataError
from ledger_rag.external.service import ExternalSession
from ledger_rag.external.tools import SCHEMAS, run_external_tool
from ledger_rag.generation.evidence import citations_from_hits
from ledger_rag.orchestration.service import RagService
from ledger_rag.security.auth import Authenticator


class SearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000, pattern=r"\S")
    as_of: date | None = None
    top_k: int = Field(default=5, ge=1, le=10)


@dataclass
class CallContext:
    service: RagService
    principal: Principal
    task: asyncio.Task[Any] | None = None
    closed: bool = False


_context: ContextVar[CallContext] = ContextVar("mcp_call_context")


def tools() -> list[Tool]:
    annotations = ToolAnnotations(
        read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True
    )
    result = [
        Tool(
            name="search_evidence",
            description="Search authorized corpus evidence, filtered before retrieval.",
            input_schema=SearchArguments.model_json_schema(),
            annotations=annotations,
        )
    ]
    for name, schema in SCHEMAS.items():
        parameters = deepcopy(schema["function"]["parameters"])
        parameters["properties"]["as_of"] = {"type": "string", "format": "date"}
        result.append(
            Tool(
                name=name,
                description=schema["function"]["description"],
                input_schema=parameters,
                annotations=annotations,
            )
        )
    return result


async def list_tools(
    ctx: ServerRequestContext[Any], params: PaginatedRequestParams | None
) -> ListToolsResult:
    return ListToolsResult(tools=tools())


async def call_tool(
    ctx: ServerRequestContext[Any], params: CallToolRequestParams
) -> CallToolResult:
    context = _context.get()
    if context.closed:
        return CallToolResult(
            content=[TextContent(type="text", text='{"error":"Request is closed"}')], is_error=True
        )
    context.task = asyncio.current_task()
    try:
        return await run_tool(context, params)
    finally:
        context.task = None


async def run_tool(context: CallContext, params: CallToolRequestParams) -> CallToolResult:
    try:
        async with request_deadline(context.service.settings.query_deadline_seconds):
            if params.name == "search_evidence":
                parsed = SearchArguments.model_validate(params.arguments or {})
                query = QueryRequest.model_validate(parsed.model_dump())
                hits, _ = await context.service.retrieve(query, context.principal)
                payload: dict[str, Any] = {
                    "citations": [c.model_dump(mode="json") for c in citations_from_hits(hits)]
                }
            elif params.name in SCHEMAS:
                data = context.service.external
                if data is None or not data.entitled(context.principal):
                    raise ExternalDataError("disabled", "External lookups are not available")
                # Each stateless request is a fresh bounded turn, never shared across users.
                arguments = dict(params.arguments or {})
                cutoff = arguments.pop("as_of", None)
                try:
                    requested = date.fromisoformat(cutoff) if isinstance(cutoff, str) else None
                except ValueError:
                    raise ExternalDataError("invalid_request", "Invalid as-of date") from None
                if cutoff is not None and not isinstance(cutoff, str):
                    raise ExternalDataError("invalid_request", "Invalid as-of date")
                session = ExternalSession(
                    data,
                    context.principal,
                    data.as_of(requested),
                    data.providers(context.principal),
                )
                name = next(name for name in SCHEMAS if name == params.name)
                citations: list[Citation] = []
                payload = await run_external_tool(name, arguments, session, citations, set())
                payload["citations"] = [c.model_dump(mode="json") for c in citations]
            else:
                raise ExternalDataError("invalid_request", "Tool is not allowed")
    except (ExternalDataError, ValidationError, TimeoutError) as error:
        message = (
            error.message
            if isinstance(error, ExternalDataError)
            else "Request deadline exceeded"
            if isinstance(error, TimeoutError)
            else "Invalid tool arguments"
        )
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({"error": message}))], is_error=True
        )
    except Exception:
        # Provider and retriever exceptions may contain URLs, documents or credentials.
        return CallToolResult(
            content=[TextContent(type="text", text='{"error":"Evidence service failed"}')],
            is_error=True,
        )
    result = CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]
    )
    if len(result.model_dump_json().encode("utf-8")) > MAX_RESPONSE_JSON_BYTES:
        return CallToolResult(
            content=[
                TextContent(
                    type="text", text='{"error":"Evidence response exceeds the size limit"}'
                )
            ],
            is_error=True,
        )
    return result


class MCPAdapter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.auth = Authenticator(
            settings.model_copy(update={"oidc_audience": settings.mcp_token_audience}),
            required_scopes=tuple(settings.mcp_required_scopes),
        )
        server: Server[Any] = Server(
            "Ledger evidence", version="0.1.0", on_list_tools=list_tools, on_call_tool=call_tool
        )
        authority = urlsplit(settings.mcp_resource_url).netloc
        self.manager = StreamableHTTPSessionManager(
            server,
            stateless=True,
            json_response=True,
            max_request_body_size=65_536,
            security_settings=TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=[authority],
                allowed_origins=settings.mcp_allowed_origins,
            ),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "resource": self.settings.mcp_resource_url,
            "authorization_servers": (
                [self.settings.oidc_issuer] if self.settings.auth_mode == "oidc" else []
            ),
            "scopes_supported": self.settings.mcp_required_scopes,
            "bearer_methods_supported": ["header"],
        }

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive)
        state = request.app.state.runtime
        started = False

        async def guarded_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            origin = request.headers.get("origin")
            if origin is not None and origin not in self.settings.mcp_allowed_origins:
                raise HTTPException(403, "Origin is not allowed")
            principal = await self.auth.authenticate(request.headers.get("authorization"))
            if "reader" not in principal.roles:
                raise HTTPException(403, "reader role required")
            state.limiter.check(principal)
            try:
                await asyncio.wait_for(state.slots.acquire(), timeout=0.01)
            except TimeoutError:
                raise HTTPException(503, "Service is at capacity") from None
            try:
                context = CallContext(state.service, principal)
                token = _context.set(context)
                try:
                    async with request_deadline(self.settings.query_deadline_seconds):
                        await self.manager.handle_request(scope, receive, guarded_send)
                finally:
                    context.closed = True
                    task = context.task
                    if task is not None and task is not asyncio.current_task():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                    _context.reset(token)
            finally:
                state.slots.release()
        except HTTPException as error:
            headers = dict(error.headers or {})
            if error.status_code == 401:
                base = self.settings.mcp_resource_url.removesuffix("/mcp")
                headers["WWW-Authenticate"] = (
                    f'Bearer resource_metadata="{base}/.well-known/oauth-protected-resource/mcp"'
                )
            await JSONResponse(
                {"detail": error.detail}, status_code=error.status_code, headers=headers
            )(scope, receive, send)
        except TimeoutError:
            if not started:
                await JSONResponse({"detail": "MCP request deadline exceeded"}, status_code=504)(
                    scope, receive, send
                )


class MCPRoute:
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        adapter = getattr(scope["app"].state, "mcp", None)
        if adapter is None:
            await JSONResponse({"detail": "Not found"}, status_code=404)(scope, receive, send)
            return
        await adapter(scope, receive, send)
