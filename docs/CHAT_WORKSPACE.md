# Chat research workspace

Ledger's browser client is a financial research desk, not a general-purpose
code-execution canvas. Its three surfaces keep the question, the answer and the
supporting evidence in view without conflating them with model reasoning.

## Run it

Requires Node 22.12+ and the existing Python backend prerequisites:

```bash
npm ci --prefix services/web --ignore-scripts
npm --prefix services/web run dev
```

Open `http://127.0.0.1:5173`. The initial **design preview** uses curated answers
over the original fictional Aurora disclosures. It makes no LLM calls, shows no
invented execution timings and does not search the web. Only its five explicit
[example questions](../services/web/src/demo.ts#L38-L45) are supported; the fifth
replays recorded SEC and FRED payloads, not a live provider call. Other preview
questions explain this limit.

For live queries, start the API with the [README instructions](../README.md#start-locally).
The development server proxies `/v1/` to `http://127.0.0.1:8080`. Set the
server-only, non-secret `LEDGER_API_ORIGIN` before starting Vite to change that
upstream. Do **not** put provider keys in any `VITE_*` variable or frontend file.
Use **Connect API** to enter the local Ledger demo token or an OIDC access token.
This is an API connection form, not an implemented identity-provider sign-in.
The first query verifies access; merely filling the form is not an auth test.

Connecting/disconnecting clears the current conversations. Access tokens and
chat state exist only in tab memory: no localStorage, sessionStorage, cookies,
cross-tab sharing or browser analytics. Reloading clears them. Live questions
and bounded conversation context are sent to the backend and its configured
model providers; "tab-only" does not mean offline inference.

## Architecture

```mermaid
flowchart LR
    Browser["React research workspace"] --> Proxy["Same-origin proxy"]
    Proxy --> Identity["FastAPI identity and admission"]
    Identity --> Graph["Request-scoped LangGraph chat"]
    Graph --> Tools["Authorized retrieval and bounded tools"]
    Tools --> Graph
    Graph --> Gate["Citation validation"]
    Graph -. "Fixed-vocabulary activity" .-> Browser
    Tools -. "Authorized citation objects" .-> Browser
    Gate --> Answer["Validated answer then done"]
    Answer --> Browser
    Graph --> Traces["Safe native OpenTelemetry spans"]
    Traces --> Collector["Allowlisted Collector to Tempo"]
    Traces -. "Optional explicitly configured sink" .-> Langfuse["Langfuse OTLP"]
```

- **React + TypeScript + Vite:** a static client for an existing Python API.
  No SSR/SEO requirement justifies adding a second full-stack application
  server. API transport is small and typed; SSE payloads receive runtime
  validation. Markdown rendering and live transport load in separate chunks.
- **LangGraph:** explicit model, bounded-tool and validation nodes for chat.
  Voice keeps its direct fast path. No checkpointer, cross-request graph state,
  durable workflow or persistent memory is claimed. See the
  [chat API contract](CHAT_API.md).
- **OpenTelemetry, with optional Langfuse:** reuse one deliberate telemetry
  path rather than double-exporting to competing vendor SDKs. LangSmith is a
  transitive LangGraph dependency, but automatic export is explicitly disabled.
  No live Langfuse integration or GenAI token/cost accounting has been validated.

## Interaction and trust contracts

| Surface | Behavior |
|---|---|
| Answer | Markdown, headings, lists, tables and fenced code. No raw HTML, executable code, iframe or remote image loading. |
| References | Numbered inline citations open an exact excerpt with title, document ID, source identifier and publication date. External SEC XBRL and FRED/ALFRED citations add a source badge and a [provenance panel](../services/web/src/components/Inspector.tsx#L170-L243) (including provider, entity, concept, form, period, value, unit, scale, as-of, vintage, retrieval time, freshness and revisions). Only credential-free HTTP(S) URLs become outbound links. |
| Artifacts | Tables and code extracted from the Markdown AST, inspectable independently and downloadable. CSV export neutralizes formula-like cells. Code is always inert text. |
| Activity | Actual server planning/retrieval/model/tool/validation events. No private chain-of-thought, simulated typing, fake progress percentages or fabricated tool traces. |
| Run details | Request ID, model, retrieval and generation/tool time, browser elapsed time, tool count and abstention status. Missing values stay missing; previews never masquerade as measurements. |
| Conversations | At most 20 per tab, 40 turns per conversation. Up to 12 bounded user/assistant messages are sent as untrusted history, never authorization or evidence. |
| Failures | Authentication, rate/capacity, malformed/truncated streams and deadlines have explicit errors. `answer` without `done` is not accepted. Stop cancels fetch; late results cannot populate another conversation. |
| Access | Tenant/group permissions come from the server identity. The UI cannot choose an arbitrary upstream URL or supply tenant claims. Provider/database/platform keys remain server-side. |

Publication cutoff means disclosure publication date, not fiscal year. The
backend planner, not the UI, may rewrite a chat follow-up into a standalone
retrieval query from screened history; history stays untrusted context
([query planning](ARCHITECTURE.md#query-planning)). The UI only sends
[bounded history](../services/web/src/contracts.ts#L49-L65) and shows the plan
digest. Citation validity is not semantic entailment. Artifact
extraction does not independently verify financial calculations or units.

There is no upload UI, persistent artifact version store, executable HTML/
Mermaid sandbox, voice UI, hosted authentication flow or multi-user conversation
database yet. These require explicit contracts and evaluation, not dead controls.

### Grounding surfaces

The `answer` event's optional `calculation`, `calculations`, `verification` and
`abstention` fields (see [CHAT_API](CHAT_API.md)) drive four surfaces. Older
payloads without them still parse; the surfaces simply do not render.

| Surface | Behavior |
|---|---|
| Calculation trace | Collapsed `<details>` per answer. Each step reads as `subtract(“USD 150 million” [C1], “USD 120 million” [C2]) = USD 30 million`. Evidence chips open the cited source and highlight the quoted span; step refs (`#0`) focus the earlier step; constants are dashed and labelled. Units, currency and scale come from the backend `unit` on each step and on the result, so a figure is never re-scaled by guesswork. The final result shows its unit and the exact backend value. |
| Verification badge | Footer pill (Verified, Unverified, Failed, Skipped) with groundedness score when present. Its disclosure lists mode, checked count, unsupported numbers and reasons; Escape or outside focus closes it. When enforce mode withheld the draft, it reports how many numbers failed (`unsupported_number_count`) without showing them. |
| Unsupported numbers | Only for non-verified, non-abstained answers: exact mentions from `unsupported_numbers` are wrapped in `<mark>` through a remark plugin, never in links or code, with screen-reader text. |
| Abstention | Calm notice with reasons, a "Thresholds uncalibrated" note when `calibrated` is false, and rephrasing tips. The trace is hidden because the backend clears the citations it references. |
| Timeline | Started/completed pairs fold into one row per operation, keyed and iconed by activity `kind` (so "Verifying grounding" is distinct from citation validation). The completed planning row expands into the plan digest: intent, route, sub-questions, companies and periods. |

The demo mode includes a verified three-step growth calculation, verified
comparison and liquidity answers, an abstained forecast question, and a labeled
offline replay of recorded SEC XBRL and FRED point-in-time evidence
([external test](../services/web/src/external.test.ts)). The demo
texts were checked against the backend number checker. Contract fixtures in
`contracts/answers.json` are produced by running the real chat pipeline
offline. Regenerate them, with the JSON Schema used by the bound-parity test, via
`uv run --frozen python scripts/dump_contract_fixtures.py` when the models change
(`--check` fails when they are stale). The wire contract, including units, activity
kinds, the plan digest and size bounds, is specified in [CHAT_API](CHAT_API.md).

## Design research and rationale

The visual direction is an original, restrained research desk: warm paper,
forest/sage accents, editorial headings, tabular financial figures and a
persistent source inspector on wide screens. Narrow screens use native modal
drawers with focus trapping and Escape/return-focus behavior.

- [Claude Artifacts](https://support.claude.com/en/articles/17153992-what-are-artifacts-and-how-do-i-use-them):
  borrow the **separation of reusable output from the transcript**, not artwork,
  branding or a claim that Ledger implements artifact versioning.
- [assistant-ui Sources](https://www.assistant-ui.com/elements/sources) and
  [AI Elements Sources](https://elements.ai-sdk.dev/components/sources):
  compact reference markers expand to evidence. Ledger retains its existing
  Citation wire contract rather than introducing a second chat protocol.
- [MDN reduced motion](https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-reduced-motion)
  and [live regions](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Guides/Live_regions):
  brief state announcements, no repeated reading of the entire transcript,
  no automatic focus theft and a user-controlled jump to latest.
- [LangGraph streaming](https://docs.langchain.com/oss/python/langgraph/streaming)
  and [Langfuse native OTEL](https://langfuse.com/integrations/native/opentelemetry):
  deliberate safe event projection and optional tracing, not private reasoning
  export or redundant instrumentation. [LangSmith self-hosting](https://docs.langchain.com/langsmith/self-hosted)
  introduces separate product/governance decisions, not a prerequisite.

Animations use short opacity/transform transitions rather than layout animation.
`prefers-reduced-motion` removes nonessential motion; state remains visible.
Bounded transcripts and memoized answer rendering precede any virtualization
dependency. The client does not invent charts from untyped numeric prose.

## Local container deployment

```bash
# Configure the backend demo token as in README before running.
docker compose --profile chat up --build -d engine web
```

Open `http://127.0.0.1:3001`. A pinned, non-root Nginx image serves the static
build and proxies `/v1/` to `engine:8080` without buffering SSE. CSP disallows inline
scripts, raw HTML execution, framing and off-origin connections. The reference
binds only to localhost. Public hosting still requires TLS, OIDC deployment,
approved provider configuration and the existing operational release gates.
Vite's dev/preview servers are **not production servers**.

Nginx JSON access logs contain method, status, duration and request ID, not
queries, token headers or user/IP identifiers. Backend metrics and redacted
logs remain the source of service-wide performance measurements.

## Reproducible checks

```bash
npm --prefix services/web run lint
npm --prefix services/web test
npm --prefix services/web run build
npm --prefix services/web run check:budget
npm --prefix services/web exec -- playwright install --with-deps chromium
npm --prefix services/web run test:e2e
# Uses the repository .venv; starts and stops isolated local API/UI processes.
npm --prefix services/web run test:live
```

Unit cases cover event framing/UTF-8, payload bounds, completion/error semantics,
history limits, reference parsing, safe links, spreadsheet exports, grounding
schema fixtures, derivation formatting and unsupported-number highlighting. Browser
cases cover desktop/mobile citation and artifact flows, download, explicit live
mode, mocked authenticated SSE/history, cancellation, truncation, secret
non-persistence, reduced motion and axe accessibility checks, plus the grounding
demo, unverified and abstained flows.

Mocked browser streams test UI contracts, not model quality or production SLOs.
The live smoke test uses the real authenticated API, same-origin proxy, graph
and fictional extractive provider; it never calls an external model. The build
gate caps total compressed JavaScript/CSS at 200 KiB and each uncompressed
JavaScript chunk at 450 KiB. These are asset budgets, not real-user latency SLOs.
The [validation record](VALIDATION.md) records actual checks separately.
For quality promotion, use [the evaluation methodology](EVALUATION.md), dev
selection and a frozen held-out test run. Never infer financial correctness
from a polished interface or a green request trace.
