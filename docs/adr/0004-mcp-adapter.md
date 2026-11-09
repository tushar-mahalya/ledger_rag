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

## Placement review (2026-11-09)

The adapter stays **inside the engine** (`services/engine`, package
`ledger_rag.mcp_adapter`), mounted at `/mcp` beside the REST and SSE routes. It is an
inbound protocol adapter of the engine, like those routes, not operator tooling
(`tools/`) and not a separate channel service. The reasons that justified splitting
voice out ([ADR 0006](0006-voice-channel-service.md)) do not apply:

| Split criterion | Voice | MCP |
|---|---|---|
| Different trust boundary or credentials | Yes: internet-facing media, speech/Twilio keys | No: same OIDC resource server, same principals |
| Different workload shape | 15-minute WebSockets, continuous audio | Short stateless HTTP requests |
| Heavy, conflicting dependencies | Pipecat, numba, onnxruntime | The `mcp` SDK, already an engine dependency |
| Different placement | Next to Twilio and speech vendors | Next to the engine |
| Needs internals beyond the public API | No | Yes: evidence search and SEC/FRED tools have no public REST endpoints |

Splitting MCP out now would add a network hop and force new public endpoints for its
tools without improving isolation. Revisit when one of these becomes true: MCP traffic
needs its own scaling or quotas, a separate authorization server or audience for
third-party agents, an independent release cadence, or tools that are not engine
capabilities.
