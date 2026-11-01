# Read-only MCP adapter

The adapter reuses `RagService.retrieve` and the SEC/FRED facade. It is a
stateless Streamable HTTP resource server at **`/mcp`**, disabled unless
`LEDGER_MCP_ENABLED=true`. It does not implement an authorization server,
token issuance, client registration, arbitrary browsing, ingestion, calculator
execution, voice or answer generation.

## Tools and isolation

`tools/list` advertises exactly three read-only tools:

1. `search_evidence`: bounded question, `top_k` (1–10) and optional `as_of`.
2. `lookup_company_facts`: the existing bounded SEC schema plus optional `as_of`.
3. `lookup_macro_series`: the existing bounded FRED schema plus optional `as_of`.

Every HTTP request authenticates independently and requires `reader`. The
verified principal is carried in request-local context; no mutable user/session
state is reused. Search accepts no tenant, role or ACL override: authenticated
tenant and group filters apply before retrieval, and returned evidence is
checked again. External calls additionally enforce the external group/tenant
allowlists. Disabled providers produce explicit errors, not alternative data.
Results include complete citable provenance. External `as_of` defaults to today
and cannot extend beyond today.

The official pinned MCP SDK handles JSON-RPC negotiation and Streamable HTTP.
The host application starts and closes its session manager in the application
lifespan; mounting alone would not start that manager. Stateless operation
avoids server-side session identity reuse and resumable event stores.

## Authentication and RFC 9728 discovery

Configure:

| Setting | Meaning |
| --- | --- |
| `LEDGER_MCP_RESOURCE_URL` | Canonical URL ending exactly in `/mcp`; HTTPS in production |
| `LEDGER_MCP_AUDIENCE` | Optional token audience override; defaults to the resource URL |
| `LEDGER_MCP_REQUIRED_SCOPES` | Required space-delimited OAuth scope tokens in the signed JWT `scope` claim |
| `LEDGER_MCP_ALLOWED_ORIGINS` | Exact bare origins allowed to access the endpoint |

Production requires OIDC. JWT signature, issuer, audience, expiry, identity
claims and configured scopes are verified. MCP audience checks are separate
from the chat API audience: an API-only token is not automatically an MCP
token. Demo mode uses the existing demo principal and bearer credential only;
it is not an OAuth deployment and has no signed scope claims.

Both `/.well-known/oauth-protected-resource/mcp` and
`/.well-known/oauth-protected-resource` serve public RFC 9728 metadata with the
canonical resource, authorization-server issuer, scopes and header bearer
method. Authentication failures advertise the path-specific metadata URL in
`WWW-Authenticate: Bearer resource_metadata="…"`. Metadata never includes
credentials. Bearer credentials in query strings are not accepted.

Origin and canonical Host allowlists protect the transport against
DNS rebinding. No wildcard hosts or origins are enabled. Reverse proxies must
preserve the configured canonical Host. Browser cross-origin CORS/preflight
support is not enabled by this adapter; deploy an explicitly configured gateway
if a browser client needs it.

## Limits and errors

HTTP requests share the API principal rate limiter and concurrency slots.
Tool work uses the query deadline and external provider budgets, breakers,
response limits and rate limits. MCP body size is limited to 64 KiB.
Each stateless external tool call is one new bounded turn; budgets do not span
separate HTTP calls. Use gateway quotas for aggregate/distributed limits.

Invalid schemas, unavailable external services and deadlines return
`isError=true` with sanitized content. Unexpected retriever/provider exceptions
return a fixed service error, without raw documents, URLs or exception details.
Cancelled work releases its concurrency slot and provider breaker probe.

## Local validation

```sh
uv run --locked pytest -q tests/unit/test_mcp_adapter.py tests/unit/test_external_chat.py
```

Tests cover protocol initialization, the three-tool allowlist, metadata and
authentication challenges, signed audience/scope/expiry checks, Host/Origin
rejection, corpus tenant/group/publication filtering, external vintage cutoffs
and concurrent principal isolation. No live tokens, deployments or external
requests are needed.

Sources: [MCP Python SDK v2.3.0 ASGI lifecycle](https://github.com/modelcontextprotocol/python-sdk/blob/v2.3.0/docs/run/asgi.md),
[MCP authorization](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization),
[RFC 9728](https://www.rfc-editor.org/rfc/rfc9728.html).
