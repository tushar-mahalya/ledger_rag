# ADR 0004: Stateless authenticated read-only MCP

- Status: accepted for the opt-in reference implementation
- Date: 2026-11-01

## Context

MCP clients need evidence access without bypassing the backend's tenant, ACL,
publication-date or external-provider controls. Session-bearing transports can
accidentally retain identities across token changes.

## Decision

Use the pinned official MCP SDK's stateless Streamable HTTP transport at
`/mcp`, with the session manager owned by the API lifespan. Authenticate every
request, verify the MCP token audience and scopes, enforce canonical Host and
Origin allowlists, and expose RFC 9728 resource metadata.

Expose only corpus search, SEC facts and FRED series. Reuse backend services
rather than building a second retrieval/provider pipeline. Request-local
principal context and shared concurrency/rate controls enforce isolation.
No mutations, voice, answer generation, arbitrary URLs or authorization-server
implementation are added.

## Consequences and release gates

The adapter is bounded and independently testable with local transports and
signed test JWTs. Stateless requests do not preserve multi-call budgets or
resumable SSE history. Aggregate quotas, browser CORS gateways, issuer/client
interoperability and deployment validation remain operator responsibilities.
See [MCP](../MCP.md) for the protocol, limits and source references.
