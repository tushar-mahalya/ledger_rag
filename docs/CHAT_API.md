# Chat API and execution activity

The authenticated `reader` endpoints are `POST /v1/query` (one JSON `Answer`)
and `POST /v1/query/stream` (SSE). Existing requests and answer/error shapes
remain supported. Both execute the same citation-validated generation path.
There is no conversation/session database, conversation cache, or resume API.

## Request and history

```json
{
  "question": "What was the operating margin?",
  "profile": "chat",
  "top_k": 5,
  "as_of": "2025-12-31",
  "history": [
    {"role": "user", "content": "Tell me about the 2024 disclosure."},
    {"role": "assistant", "content": "Revenue was USD 120 million. [C1]"}
  ]
}
```

- `history` is optional and defaults to `[]`; `null` is not accepted.
  At most **12 messages**, **8,000 characters per content string**, and
  **32,000 content characters total** are accepted. Empty content is allowed.
  Roles are exactly `user` or `assistant`; extra message fields are rejected.
- Nonempty history with `profile: "voice"` is rejected with HTTP 422.
  Voice with omitted/empty history retains its bounded, tool-free fast path
  and existing rerank bypass.
- History is **untrusted context, not evidence or authorization**. It is
  serialized as a separate conversation-history JSON message, not replayed as
  system messages or trusted assistant claims. The current question/profile and
  current retrieval evidence are a separate final user message. Historical
  citation IDs do not establish support for the current answer.
- With `LEDGER_PLANNER_ENABLED=true` (default), chat follow-ups that are not
  self-contained are rewritten into a standalone **retrieval** query (see
  [query planning](ARCHITECTURE.md#query-planning)). The rewrite reads only
  history JSON as untrusted data, may not add periods or URLs, and changes only
  the search text; filters still come from the principal. The model still
  receives the original question. Voice and `planner_enabled=false` retrieve with
  the current question only. Chat can request a bounded corrective search. That search changes only the
  question and retains the authenticated principal, ACLs, `as_of`, and `top_k`.
  `as_of` limits **publication date**, not financial reporting period.
- The caller supplies any history on each request. Nothing is saved globally.

Question length, authentication, rate limits, body limits, deadlines, and
concurrency limits are unchanged. Authentication and request validation failures
occur before the SSE response and retain ordinary HTTP error statuses.

## SSE wire contract

Frames use `event: NAME`, `data: JSON`, and a blank line. Clients must tolerate
arbitrary transport chunk boundaries and ignore event names they do not support.
These are lifecycle events, **not token streaming or private reasoning**.

### `activity` (additive)

```text
event: activity
data: {"stage":"generation","status":"started","label":"Generating answer"}

```

The public schema is:

```text
{
  stage: "retrieval" | "planning" | "generation" | "tool" | "validation",
  status: "started" | "completed",
  label: <fixed label below>,
  tool?: "calculator" | "search_evidence"
}
```

| Stage / operation | Fixed label | Optional `tool` |
| --- | --- | --- |
| retrieval | `Retrieving evidence` | absent |
| planning | `Planning query` | absent |
| generation | `Generating answer` | absent |
| calculator | `Calculating` | `calculator` |
| corrective search | `Searching evidence` | `search_evidence` |
| rejected tool request | `Checking tool request` | absent |
| validation | `Validating citations` | absent |

`started` is emitted on entering the actual operation; `completed` only on
normal exit. There are no simulated steps or artificial delays. An exception,
timeout, or cancellation can leave an operation without `completed`.
Handled calculator errors and rejected tool requests can complete normally:
**completed means the operation finished, not that an answer is correct**.
Likewise, completed citation validation can produce an abstention.

Typical ordering is retrieval started/completed, `evidence`, planning
started/completed (chat graph entry; absent when the planner is disabled), generation
started/completed, validation started/completed, `answer`, `done`. Each executed
tool adds its own started/completed pair followed by another model operation.
`search_evidence` contains a nested retrieval lifecycle. Over-budget tool batches
are not executed and do not create tool activity. With no evidence and no search
authority, direct service callers skip generation and only validate abstention.
A confident out-of-scope plan (for example prompt-injection or investment-advice
requests) skips both retrieval and generation and abstains after validation.

Activities contain no questions, history, prompts, tool arguments, tool call IDs,
raw tool results, provider hidden-reasoning fields, or exception messages.
Do not label these records as the model's thoughts or expose LangGraph state.

### `evidence`

```text
{
  request_id: string,
  retrieval_ms: number,
  sources: string[],
  citations: Citation[]
}
```

The first three fields are unchanged; `citations` is additive and contains the
initial authorized retrieval evidence. Citation IDs use the same deduplicated
assignment as generation:

```text
Citation = {
  id: string, document_id: string, source: string, title: string,
  published_at: "YYYY-MM-DD", quote: string
}
```

Evidence is a retrieval preview, not an approved answer. Corrective search may
add citations later; the final answer's citations are authoritative for the
answer. Citation quotes are intentionally exposed authorized evidence, unlike
private tool protocol messages.

### `answer`, `done`, `error` (unchanged)

`answer` carries the existing `Answer` JSON:

```text
{
  answer: string, citations: Citation[], abstained: boolean,
  profile: "chat" | "voice", retrieval_ms: number, generation_ms: number,
  request_id: string, model: string, tool_calls: integer
}
```

No answer is emitted before citation validation. Missing, unknown, or
paragraph-level missing citation markers produce the existing explicit
abstention and empty citations. `done` carries `{}` after success, including
successful abstention.

After a stream has begun, failures produce `error` with exactly:

```json
{
  "detail": "Stream failed; do not treat partial events as a final answer",
  "request_id": "<request ID>"
}
```

There is no `done` on this path. Clients must handle abrupt transport termination
without `done` as incomplete as well. Use `AbortController`/connection closure to
cancel: the producer, pending queue read, in-flight graph/provider/retrieval work,
and concurrency permit are cleaned up. The stream buffer is bounded to **16
frames**, applies backpressure, and uses task completion rather than a sentinel
that could block cleanup on a full queue. No heartbeat or reconnect/resume
protocol is implemented.

## Graph, budgets, and compatibility

Chat uses a typed LangGraph `StateGraph`:

```text
START -> model -> tools -> model -> ... -> validation -> END
            \----------------------------^
```

The conditional entry can skip the model for an evidence-free, search-disabled
direct call. Model, tool execution, and citation validation are separate nodes,
not a graph wrapper around a manual agent loop. `max_tool_rounds` retains its
existing meaning: a **total tool-call budget** (default 3, configured range 1–5),
including rejected requests, not unlimited calls per round. An oversized batch
is rejected whole. At most budget + 1 model calls run, with an independent graph
recursion bound. Tools are sequential, allowlisted, bounded and read-only.
Corrective evidence is deduplicated and capped at 12 citations.

The compiled graph has no checkpointer, store, cache, or request-specific global
state. Each invocation owns its messages, citations, counters, query/principal
snapshot, and async activity callback. Voice invokes the existing provider
contract directly without entering the chat graph. Existing positional
`RagService.generate(..., principal=None)` callers remain compatible; `on_activity`
is an optional keyword-only async callback on `generate` and `retrieve`.
Callback exceptions propagate; callbacks participate in cancellation/deadlines.
Direct service callers still supply their own deadline, as before; HTTP and
voice entry points enforce their existing deadlines.

## Measurement and tracing

- `request_id` matches `X-Request-ID` and links responses with request logs;
  `model`, `retrieval_ms`, `generation_ms`, and `tool_calls` remain available.
  Answer timings are rounded to three decimals. `retrieval_ms` measures initial
  retrieval; corrective searches are part of `generation_ms`. Generation
  includes graph/tool/validation overhead. Callback/backpressure time can be
  included in phase timings; these are not model-only latency measurements.
- Native OTEL spans include `rag.retrieve`, `rag.plan` (inside retrieval),
  `rag.generate`, `rag.chat.plan`, `rag.chat.model`, `rag.chat.tools`,
  `rag.chat.validation`, and per-tool `rag.tool`. Plan spans carry only enum,
  count, boolean and timing `rag.plan.*` attributes (route, decision source and
  confidence, operation, rewrite method, sub-query/company/year/metric counts),
  never the question, history, rewritten query or entity values.
  They retain nesting and timing, with fixed stage/tool identifiers, model name,
  counts, profile, and abstention metadata. New spans and the HTTP root record
  error status without capturing exception text, stack traces, or input/output.
- Existing Prometheus retrieval/generation latency and abstention metrics remain.
  There is no fabricated token usage, monetary cost, semantic score, or claim of
  full GenAI semantic-convention instrumentation.
- OTLP remains disabled unless `LEDGER_OTLP_ENDPOINT` is configured. LangGraph
  transitively installs LangSmith, but automatic LangSmith tracing is explicitly
  disabled for these invocations, even if tracing environment flags are set.
  No Langfuse SDK, tracer callback, graph-state export, or paid call was added.

### Optional existing Langfuse OTLP route

The existing exporter can be configured with
`LEDGER_OTLP_ENDPOINT=https://<your-langfuse-host>/api/public/otel`; it appends
`/v1/traces`. Use protected/Vault-rendered `LEDGER_OTLP_HEADERS` JSON containing
`Authorization: Basic <base64(public-key:secret-key)>` and, for v4 ingestion,
`x-langfuse-ingestion-version: "4"`. Choose the correct regional or self-hosted
endpoint. Do not put credentials in URLs, source control, logs, or browser code.
This is an **optional configuration route**, not a verified live Langfuse
deployment or a full Langfuse generation/evaluation integration.

The checked-in `deploy/otel-collector.yaml` retains `http.request.method`,
`http.response.status_code`, `http.route`, and `error.type`, plus these safe
fixed-enum, count, and boolean span attributes:
`rag.stage`, `rag.tool`, `rag.profile`, `rag.tool_calls`, `rag.abstained`,
`rag.citation_count`, `retrieval.hit_count`, and the `rag.plan.*` keys listed
above. Collector-routed traces therefore
retain this metadata alongside span structure, timing, and status.
`gen_ai.request.model` remains excluded because configured model strings are not
sanitized. Queries, prompts, history, tool arguments, and raw tool results are not
allowlisted. This does not establish full GenAI instrumentation or verify a live
Langfuse deployment.

## Verification and limits

Focused offline tests cover actual compiled-graph invocation, node spans,
real callback timing, nested corrective retrieval, principal/history isolation,
budgets, malformed/rejected tools, citation abstention, history validation,
failure sanitization, bounded buffering, disconnect/deadline cleanup, and tracing
disablement. Run:

```bash
uv run ruff check src tests
uv run mypy
uv run pytest tests/unit/test_chat_graph.py tests/integration/test_chat_stream.py
```

These use deterministic providers/local retrieval, not paid models or external
tracing. Citation-marker validation is **not semantic entailment or financial
correctness verification**. History isolation tests do not prove an LLM immune
to prompt injection. The graph does not add durable memory, checkpoint replay,
human approval, token streaming, full GenAI telemetry, automatic evaluations,
or measured model-quality improvements. Existing evaluation tooling remains
separate; no benchmark score or production performance result is claimed here.

Official references consulted through Context7:
[LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api),
[Graph API usage](https://docs.langchain.com/oss/python/langgraph/use-graph-api),
[LangGraph observability](https://docs.langchain.com/oss/python/langgraph/observability),
[Langfuse native OpenTelemetry](https://langfuse.com/integrations/native/opentelemetry).
