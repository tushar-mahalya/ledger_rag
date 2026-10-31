---
name: ledger-generation
description: "Use when changing OpenAI/Anthropic adapters, chat tool calling, grounding, citations, abstention, request deadlines or tenant authorization in Ledger RAG."
---

# Grounded generation and bounded tools

## Read

[Service](../../../src/ledger_rag/orchestration/service.py),
[providers](../../../src/ledger_rag/generation/providers.py),
[tools](../../../src/ledger_rag/generation/tools.py),
[auth](../../../src/ledger_rag/security/auth.py),
[API](../../../src/ledger_rag/api/app.py).

## Procedure

1. Identify the protocol and execution profile. OpenAI chat-completions and
   Anthropic messages compatibility does not imply Responses API compatibility
   or that every model supports tools.
2. Trace the contract from auth to retrieval, tool loop, answer validation and
   response. Derive tenant/groups from signed identity, never request metadata.
3. Keep retrieved text/tool output out of instruction authority. Test hostile
   document text, unknown citations, fabricated tool IDs and malformed arguments.
4. Preserve allowlisted tools, validated schemas, Decimal arithmetic bounds,
   cumulative call/round budgets and an overall deadline. No `eval`, shell tool
   or autonomous network fallback.
5. Fail explicitly on upstream/protocol errors, without leaking provider bodies.
   Cancellation must release permits; do not retry mutations or silently switch
   providers with different privacy/accuracy properties.
6. Validate citations before emitting the final answer or synthesizing speech.
   Test empty evidence, conflicting periods, unsupported answers and abstention.
7. For caching, include tenant/ACL, corpus/model/prompt version and as-of boundary.
   Never assume similar wording permits reuse across tenants or amended filings.
   Caching is a proposed track, not an existing cache to depend on.

```bash
uv run --locked pytest -q tests/unit/test_providers.py tests/unit/test_core.py tests/integration/test_api.py tests/integration/test_resilience.py
```

## Done / stop

Show protocol fixtures and failure-path tests for the changed behavior. Stop if
the feature requires weakening grounding/auth or speaking unvalidated text.
SSE currently sends evidence followed by a validated complete answer, not tokens.
