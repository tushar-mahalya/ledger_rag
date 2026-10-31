# Architecture: Reusable RAG Backend for Financial Disclosures

**Status key used throughout this document:**
- **[IMPLEMENTED]** — exists in this repo's current bounded reference build, as reported by the implementation workstreams and reflected here without embellishment.
- **[TARGET]** — a production-grade goal informed by RESEARCH.md, not yet built in this repo.
- **[FUTURE]** — an evidence-gated upgrade path (e.g., graph RAG), intentionally deferred.

This document is descriptive of a design and, where marked [IMPLEMENTED], of code built by parallel implementation work in this project. It does not itself constitute a commitment to ship [TARGET]/[FUTURE] items without further scoping.

---

## 1. Design principles (derived from RESEARCH.md)

1. **No vendor lock-in at the model layer.** OpenAI- and Anthropic-style endpoints are accessed behind a provider interface; the bounded reference ships a zero-key local demo provider plus an optional real-provider path. Model family names are read from configuration/live docs, never hardcoded (RESEARCH.md §4, §7.8).
2. **Hybrid retrieval by default, graph RAG only when evidence demands it.** Dense+BM25+RRF+rerank is the baseline (RESEARCH.md §2); GraphRAG/LightRAG/HippoRAG are [FUTURE], gated on a measured multi-hop-query failure-rate ablation (RESEARCH.md §3, §6.5), not adopted speculatively.
3. **Untrusted-content discipline.** Per OWASP LLM01 (RESEARCH.md §4), retrieved chunk text is treated as untrusted data passed to the generator, not as instructions — this applies to both the extractive demo path and any future agentic tool-calling path.
4. **License-aware data handling.** SEC EDGAR access is free and unrestricted, but **issuer-authored filing text is not public domain** (17 U.S.C. §105 covers only government-authored works, not privately-authored filings merely hosted on EDGAR — RESEARCH.md §1.1 correction); filing narrative text is fetched live and cited in small excerpts rather than bulk-vendored. Only factual/XBRL data and the explicitly-licensed dataset subsets (FinanceBench 150-sample under CC-BY-NC-4.0, TAT-QA under CC-BY-4.0, FinQA/ConvFinQA/DocFinQA under MIT/CC-BY-4.0 per their publishers) are used per their stated terms; FinDER and the full FinanceBench set are NC-licensed and used only within fair eval-harness bounds, never vendored as copied text (RESEARCH.md §1).
5. **Independent indexing from serving.** Ingestion/embedding/reranking are workers, decoupled from the request-serving API, so that indexing latency (seconds-minutes) never blocks interactive chat latency targets (RESEARCH.md §5).
6. **Secrets never touch the shell.** Vault/file-based secret injection only; no `.env` sourcing or shell-exported credentials (see §5 below).

---

## 2. High-level topology

```
                      ┌───────────────────────────────┐
                      │        API (modular monolith)  │
                      │  FastAPI-style request surface  │
                      │  - chat / query endpoints        │
                      │  - auth (OIDC) + ACL enforcement  │
                      │  - provider router (LLM/embed)    │
                      │  - corrective-retrieval tool loop │
                      └───────────┬───────────┬─────────┘
                                  │           │
                 ┌────────────────┘           └───────────────┐
                 ▼                                             ▼
   ┌─────────────────────────┐                    ┌─────────────────────────┐
   │ Indexing worker(s)        │                    │ Model/provider workers   │
   │ - chunk (parent/child)     │                    │ - embedding calls         │
   │ - contextual augmentation  │                    │ - LLM chat calls          │
   │ - embed + upsert to Qdrant │                    │ - optional rerank (TEI)   │
   └────────────┬───────────────┘                    └────────────┬────────────┘
                │                                                  │
                ▼                                                  ▼
   ┌─────────────────────┐   ┌─────────────────┐   ┌───────────────────────┐
   │ Qdrant (optional)     │   │ Postgres          │   │ Redis                  │
   │ dense+sparse vectors,  │   │ - ACL / RLS        │   │ - cache / rate-limit    │
   │ RRF fusion, payload     │   │ - outbox pattern    │   │ - ephemeral session     │
   │ filter (tenant/ACL/date)│   │ - audit/eval records │   │   state                 │
   └─────────────────────┘   └─────────────────┘   └───────────────────────┘

Cross-cutting: OIDC (authn) · Vault/file-secrets (secret delivery) · OpenTelemetry (traces/metrics)
              · Grafana/Loki/Tempo (observability) · uv (Python env/dep management)
```

This is a **modular monolith**, not microservices: one deployable API process, but with indexing and model-calling logic isolated into independently schedulable workers/modules so they can be scaled, rate-limited, or swapped (e.g., local demo provider → real OpenAI/Anthropic provider) without touching request-serving code. This matches RESEARCH.md §5's requirement that indexing never blocks the request path, and keeps operational surface area small enough for a "bounded reference" to remain auditable.

---

## 3. Retrieval pipeline design

**[TARGET] full pipeline:**
1. Ingest SEC filing (or other licensed source) → parent-section + child-sentence/table-row chunking (RESEARCH.md §2, LlamaIndex pattern).
2. Contextual augmentation: prepend LLM-generated chunk context (company/filing/period/line-item) per Anthropic's Contextual Retrieval methodology (RESEARCH.md §2) — [FUTURE] relative to current build; not yet implemented.
3. Embed (dense + sparse) and upsert into Qdrant with tenant/ACL/as-of-date payload fields.
4. Query time: hybrid dense+BM25 retrieval, RRF fusion, optional cross-encoder rerank, top-k passed to the generator with citations.
5. Generation: provider-routed LLM call; answer must cite retrieved chunk IDs; untrusted chunk content is never treated as instructions (OWASP LLM01).

**[IMPLEMENTED] in the current bounded reference, as reported by the implementation workstream:**
- A **local in-memory demo path**: lexical (keyword/BM25-style) "core memory" search requiring no embeddings, no GPU, and no API keys — this is the zero-dependency default so the repo runs out of the box.
- An **optional real-provider path** using a real `AsyncQdrantClient(':memory:')` (and, in deployment, a real Qdrant server) performing actual **dense+sparse hybrid search with RRF fusion**, with tenant/public ACL filtering and `as_of` date filtering, covered by a real-SDK integration test suite (not mocked) that is actively growing — exact test counts are intentionally omitted here since they change frequently during active development; verify current counts via the repo's own CI output rather than this document — with **mypy strict** and **ruff** passing as of the last implementer report.
- **Document replacement semantics**: upsert treats an incoming document's chunk set as authoritative — new chunks are upserted first, then stale chunks for that tenant/document are deleted via a `HasId`-exclusion filter. This is **explicitly non-transactional**: there is no Qdrant compare-and-swap primitive, so same-document concurrent writes must be externally serialized (single-writer per document, or an application-level write lock) — documented here as a known operational constraint, not a bug to silently work around. In-process, same-instance document updates are serialized with a write lock; **cross-worker/cross-process replacement is not yet transactional** and requires a deployment-level single-writer convention — **[TARGET]**, not yet solved generally.
- **Chunk ID discipline**: chunk IDs are derived via `UUID5` from caller-supplied deterministic identifiers; an ownership preflight check **rejects** attempts to write a chunk ID that collides with an existing chunk owned by a different tenant/document. Callers (indexing workers) are responsible for namespacing deterministic chunk IDs by tenant + document to avoid collisions — this is a caller contract, not an automatic guarantee.
- **Collection metadata / version gating**: Qdrant collection identity/metadata requires a modern SDK/server (Qdrant v1.20.0 line, RESEARCH.md §4). The client **fails closed** — refuses to operate — against older or unidentified/unrecognized collections, rather than guessing compatibility. This is a deliberate safety choice over silently degrading.
- **BM25 scoring detail**: the sparse/BM25 term-frequency component uses a **fixed reference average-document-length (avgdl = 256)** combined with Qdrant's corpus-level IDF, rather than dynamically recomputing avgdl from the live corpus. This is a known, documented approximation, not a hidden defect — it should be revisited if corpus document-length distribution shifts significantly from the assumed baseline.
- **No token streaming** — the current implementation returns complete responses only; streaming generation is **[TARGET]**, not implemented.
- **HTTP error handling**: provider/Qdrant HTTP errors are sanitized before surfacing (no raw upstream error leakage to clients) and are **not automatically retried** — retry/backoff policy is **[TARGET]**.
- **Reranking**: an **optional TEI (Text Embeddings Inference) cross-encoder `/rerank` service wrapper** is implemented — chat-path candidates (10) are reranked down to the configured top-k; the voice/IVR tool path **bypasses reranking** (latency-sensitive, top-k capped at ≤3 for voice — see §4).
- **No graph-based retrieval and no late-interaction (ColBERT/ColPali) retrieval are implemented** in the current build — these remain **[FUTURE]** per the evidence-gated upgrade policy in RESEARCH.md §3.
- An **eval harness** exists for retrieval/generation scoring, separate from the serving path — implements the general shape of RESEARCH.md §6's protocol, exact dataset wiring not independently audited in this document.
- **Test/verification status (per implementer report):** a passing adapter test suite (exact count intentionally not tracked in this document — it evolves with active development; treat the repo's CI as the source of truth for current counts), with **mypy strict** applied including test files, and **ruff** passing. Real-SDK integration tests additionally prove: inclusive `as_of` date-boundary semantics, public/overlap ACL-group matching, foreign-tenant same-document chunk IDs are preserved (not clobbered) across a replacement operation, and ACL scope correctly tightens after a replacement (no stale broader grant survives). The `search_evidence` tool's conversational history handling was tested across both the demo/extractive and real-Qdrant protocols.
- **Failure-safety for non-transactional replacement (refines §3's caveat above):** an **instance-level operation lock** serializes reads and writes against a given collection/document. A mutation sets an **"uncertain" flag** *before* issuing the Qdrant write, and only clears it after all associated stale-chunk deletes succeed. If a mutation fails or is cancelled while uncertain, the instance is **quarantined**: subsequent `search` and `upsert` calls raise a reconciliation-required error and report `ready=false` rather than silently serving a possibly-inconsistent state. **A process restart is explicitly NOT a safe substitute for reconciliation** — the quarantine state must be cleared by an explicit reconciliation procedure, not assumed to self-heal on restart. Cross-process writes still require external single-writer coordination (unchanged from §3).

---

## 4. Chat / agentic tool layer

**[IMPLEMENTED]:** a bounded chat loop with two tools:
- A **calculator tool**, useful for the numerical-reasoning failure mode that FinQA/TAT-QA explicitly identify as dominant (RESEARCH.md §2) — arithmetic is delegated to a deterministic tool rather than trusted to free-form LLM arithmetic.
- A **`search_evidence` corrective-retrieval tool** that preserves the original request's tenant, group/ACL scope, and `as_of` date across tool calls (so a tool-driven follow-up query cannot escape the caller's authorization scope), with retrieved citations capped at 12 per response to bound prompt size and keep citation review tractable.

This is a lightweight, prompt-level analogue of the Self-RAG/CRAG *pattern* (explicit evidence-check-before-answer, RESEARCH.md §3) implemented via tool-calling rather than a fine-tuned reflection-token model — chosen specifically because fine-tuning a dedicated model is out of scope for a provider-agnostic reference build (RESEARCH.md §3, Self-RAG row).

**Voice/IVR path [IMPLEMENTED, partial]:** voice invocations **disable** the tool set described above and cap retrieval to **top-k ≤ 3** with reranking bypassed — a deliberate latency-over-recall tradeoff aimed at the ~500–800ms conversational-turn budget documented in RESEARCH.md §5. **No actual STT/TTS/telephony integration is implemented** — this is a latency-budget-aware code path, not a working IVR system; real-time audio I/O remains **[TARGET]**.

**Extractive demo provider contract [IMPLEMENTED]:** the local demo provider parses evidence directly out of the user message rather than performing real retrieval, to allow deterministic zero-key testing. Expected input shape: a JSON value in the user message that is either a **list** of evidence items or a single **object** keyed by `chunks` / `evidence` / `context`; each item carries a chunk `text` (or `quote`) field and an optional citation `id`. This is a **test/demo harness contract**, not the production retrieval interface — real deployments route through the Qdrant-backed hybrid retrieval path instead.

- **Lexical extraction caveat [IMPLEMENTED]:** the extractive demo's quote-selection logic is a **conservative, lexical sentence-matching heuristic** (period/year tokens, basic title/context cues), not a semantic understanding of the question. It carries **no general semantic-faithfulness guarantee** — passing a specific phrasing is evidence for that case, not proof the heuristic generalizes to arbitrary phrasing. A prior regression (comparative sentences containing a requested year alongside a prior-period year were being incorrectly rejected or clause-split, losing period/value attribution) has been fixed per implementer report: whole comparative sentences are now preserved when the requested year appears in them; yearless sentences may still rely on a sole matching document-title year even when other years appear elsewhere in the chunk; publication dates are still never used to infer reporting period; sentences/documents covering only a prior, non-requested period are still excluded for explicit single-year questions. This was validated against the project's own smoke evaluation run (`uv run python -m ledger_rag.evaluate --smoke --repeats 2`, per implementer report) reporting abstention-specificity, answer-containment, citation-document-precision, recall, and MRR all at 1.0 on that smoke set — **a smoke-set result, not a claim of general-case correctness**; exact test counts are intentionally not pinned down here (see test-count note above). Precise internal matching rules remain an implementation detail best consulted directly in code/tests rather than fully restated here, since this behavior has already changed once during this session. This heuristic applies only to the zero-key demo/extractive path — it is separate from the real Qdrant-backed retrieval path described below.

---

## 5. Configuration, secrets, and identity

**[IMPLEMENTED] convention (as specified by the implementation workstream and reflected here, not altered):**
- Application settings use an `LEDGER_` environment-variable prefix (`Settings(..., env_prefix="LEDGER_")`), with secret values loaded from files under a configurable `LEDGER_SECRETS_DIR` (default `/run/ledger-secrets` in the reference compose setup) via a `Settings(_secrets_dir=...)` mechanism — i.e., secret values are read from mounted files, not from process environment variables or shell-sourced `.env` files.
- Expected Vault-delivered secret filenames: `LEDGER_LLM_API_KEY`, `LEDGER_EMBEDDING_API_KEY`, `LEDGER_QDRANT_API_KEY` (one file per secret, filename matching the corresponding env-style key name).
- Production container startup is expected to **wait for secret files to materialize** (`deploy/wait-for-secrets.py`) before `exec`-ing the API process (`uvicorn`) — no shell-sourcing of secrets at any point, consistent with the "secrets never touch the shell" principle (§1.6).
- **Validator requirement (as specified to the parent config owner):** an empty `SecretStr` (empty string) must be **rejected by the settings validator**, not only a `None`/missing value — this closes a gap where a demo/ops-compose default like `LEDGER_DEMO_TOKEN=${LEDGER_DEMO_TOKEN:-}` could otherwise silently produce an empty-but-"present" secret in production, defeating the Vault-file-injection path. Per a later implementer update, the demo-token validator now also enforces a **minimum length of 8 characters** (not just non-empty), tightening the earlier bare-non-empty check. This is recorded here as a documented requirement on the configuration module; whether it is currently enforced in code is **owned by the configuration implementation, not verified independently in this document.**

**[TARGET], not yet built:** full Vault dynamic-secret leasing/rotation (vs. static file mount), OpenBao as the OSI-licensed alternative given HashiCorp's BSL relicensing of Vault (RESEARCH.md §4) — worth evaluating before a production commitment to either, since BSL restricts competing hosted offerings and converts to open terms only after a multi-year delay.

**Identity/authz [TARGET]:** OIDC for authentication (RESEARCH.md §4, openid.net Core 1.0) feeding into Postgres Row-Level Security policies (`SET LOCAL app.tenant_id` + `CREATE POLICY ... USING (...)`, `FORCE ROW LEVEL SECURITY`) for per-tenant document ACL enforcement, mirrored by Qdrant payload-level tenant/group filters at the vector-search layer (already [IMPLEMENTED] at the Qdrant layer per §3; the Postgres RLS layer for broader application data is [TARGET]).

---

## 6. Data layer

| Store | Role | Status |
|---|---|---|
| **Qdrant** (optional) | Dense+sparse vector search, RRF fusion, tenant/ACL/as-of payload filtering, document replacement | [IMPLEMENTED] per §3, with noted non-transactional-replacement caveat |
| **Postgres** | System of record: documents, chunks metadata, ACL grants (via RLS), outbox events for downstream notification/indexing triggers, audit log, eval-run records | [TARGET] — outbox pattern (microservices.io canonical description, RESEARCH.md §4) chosen specifically to guarantee atomicity between "document ingested" state changes and the event that triggers async (re-)indexing, without a distributed transaction across Postgres and the indexing worker queue |
| **Redis** (or Valkey) | Response/embedding cache, rate limiting, ephemeral session/tool-loop state | [TARGET]. Given Redis's 2024-2025 licensing changes (RSALv2/SSPLv1/AGPLv3, no longer plain BSD — RESEARCH.md §4), **Valkey (BSD-3, Linux Foundation)** should be the default evaluated option for a reference build that wants to remain unambiguously permissively licensed; either is wire-compatible enough for this role. |

---

## 7. Observability and operations

- **OpenTelemetry [IMPLEMENTED]:** OTLP-exported distributed traces and Prometheus-scraped metrics are wired into the API/worker paths per the implementer report. Log export is also implemented (note: RESEARCH.md §4 flags that OTel's Python **Logs** API/SDK is officially still non-GA upstream — this project's log pipeline should be understood as using that still-maturing upstream surface, not as evidence the upstream spec itself has reached GA; do not cite this implementation as proof of OTel Python Logs GA status). The surrounding **operations profile (dashboards, alerting, retention policy) is still being wired** — [TARGET]/in progress, not yet a complete operational stack.
- **Grafana/Loki/Tempo/Mimir [TARGET], in progress:** the consuming dashboard/log-aggregation/trace-storage stack for the OTLP/Prometheus signals above is part of the operations profile being wired (previous bullet) and is not yet a documented, complete deployment in this repo. Note all of Grafana/Loki/Tempo/Mimir are AGPLv3-licensed since their 2021/2022 relicensing (RESEARCH.md §4); acceptable for internal/self-hosted ops tooling, but flagged here since AGPLv3 carries copyleft obligations that differ from the MIT/CC-BY-licensed data/eval assets discussed in RESEARCH.md §1 — do not conflate observability-stack licensing with dataset licensing in any compliance review.
- **uv** [IMPLEMENTED expectation]: Python dependency/environment management for this repo is expected to use `uv` (lockfile-based, single-binary, replaces pip/venv/pyenv workflows per Astral's official docs, RESEARCH.md §4) rather than ad hoc pip/venv — consistent with the "no unsafe dotenv or shell sourcing" operational principle already applied to secrets (§5).

---

## 8. Threat model summary (informs, does not replace, a full security review)

Derived directly from RESEARCH.md §4's OWASP citations:
1. **Broken Object/Function-Level Authorization (OWASP API1/API5)** — every retrieval and chat-tool call must re-check the caller's tenant/ACL scope server-side; the `search_evidence` tool's preservation of the original request's tenant/group/as_of scope (§4) is a concrete, already-implemented mitigation for the specific risk of a tool call escaping its caller's authorization.
2. **Excessive Data Exposure / Mass Assignment (API3)** — citation caps (12 for chat) and payload-filtered retrieval limit how much of the underlying corpus a single response can expose, but this is a soft mitigation, not an authorization boundary; the real boundary is the ACL filter enforced at the Qdrant/Postgres layer, not the cap.
3. **Prompt Injection via retrieved content (LLM01)** — retrieved financial-filing text (including footnotes, which can contain adversarially-crafted or simply misleading natural language in real filings) must be treated as untrusted data in the generation prompt, never as instructions; this applies to both the demo extractive provider and any future real retrieval path.
4. **Non-transactional document replacement (§3)** as an availability/integrity risk, not just a performance note: concurrent writers to the same document without external serialization can produce a transient state with both old and new chunks, or missing chunks, visible to a concurrent reader — this must be called out explicitly to any production deployment plan, not silently accepted.
5. **Secret-handling** (§5): file-mounted Vault secrets + startup-blocking wait-for-secrets + rejection of empty SecretStr values are the concrete mitigations against the common failure mode of a service silently starting with no/empty credentials and failing open or masking a misconfiguration as a "public demo mode."

---

## 9. Explicitly deferred (not in this bounded reference)

Per RESEARCH.md's evidence-gated-upgrade philosophy, the following are intentionally **not** built, and should not be assumed present:
- GraphRAG / LightRAG / HippoRAG graph-based retrieval (§3 of RESEARCH.md) — gated on a measured multi-hop-query ablation.
- ColBERT/ColPali late-interaction or vision-based table retrieval — requires GPU/VLM infrastructure, explicitly out of scope for the no-GPU/no-keys bounded reference.
- Anthropic-style Contextual Retrieval chunk augmentation at indexing time — a [TARGET] identified as high-value (49%/67% reported failure-rate reduction, RESEARCH.md §2) but not yet implemented.
- Direct SIP/RTP termination, neural VAD, LLM token streaming into TTS, and executing the offered agent transfer (a Twilio `<Dial>` redirect) — the streaming voice gateway itself is [IMPLEMENTED], see §10.
- Token streaming, automatic HTTP retry/backoff, cross-process transactional document replacement, full OIDC/Postgres-RLS stack, a complete Grafana dashboard/alerting layer atop the already-implemented OTel/Prometheus signal export, Vault dynamic leasing.
- **Clarification:** real Qdrant-backed hybrid retrieval and the response **citation structure** (chunk IDs/evidence attached to answers) are **already [IMPLEMENTED]**, per §3/§4 above — these are not on this deferred list. What remains deferred is a **semantic faithfulness verifier** (an explicit check that a generated answer is actually, semantically supported by its cited citations, beyond the citation structure merely existing) — that verification layer is [TARGET]/[FUTURE], distinct from the already-implemented act of attaching citations.

This list exists specifically so that this document cannot be read as claiming more capability than is actually built — per the user's explicit requirement to distinguish implemented baseline from future production integrations.

---

## 10. Voice gateway [IMPLEMENTED]

Package `ledger_rag.voice` (disabled unless `LEDGER_VOICE_ENABLED=true`; endpoints close with 1008 otherwise). Pure-Python G.711 mu-law, resampling and WAV handling (no `audioop`, which Python 3.13 removed), an energy VAD, OpenAI-compatible STT/TTS adapters (OpenAI, Groq STT, speaches / faster-whisper-server, Kokoro-FastAPI) and a per-call `VoiceSession` state machine.

```
 Caller ── PSTN ── Twilio <Connect><Stream> ──wss── /v1/voice/twilio      (/v1/voice/stream: PCM16 16 kHz + JSON)
                                                     │ base64 mu-law 8 kHz, 20 ms frames
                                                     ▼
 LISTENING ── EnergyVad (20 ms RMS frames, 500 ms trailing silence, 15 s cap) ── end of utterance
     ▲                                               ▼
     │                         TRANSCRIBING  POST {stt}/audio/transcriptions (WAV, span voice.stt)
     │                                               ▼
     │                         THINKING      RagService.retrieve + generate (Profile.VOICE, caller principal,
     │                                       citation-validated; abstention -> polite transfer offer)
     │                                               ▼
     │ mark echoed /           SPEAKING      POST {tts}/audio/speech, streamed (span voice.tts)
     │ barge-in "clear"                      24 kHz PCM/WAV -> 8 kHz -> mu-law 160-byte media frames + mark
     └───────────────────────────────────────────────┘
```

**Latency budget** (targets for a well-provisioned deployment, not measured guarantees; every turn reports actuals via `turn_metrics`, Prometheus `ledger_voice_*` histograms, structured logs and the `voice.turn`/`voice.stt`/`voice.tts` spans):

| Stage | Budget | Notes |
|---|---|---|
| VAD endpoint (trailing silence) | ~500 ms | `LEDGER_VAD_TRAILING_SILENCE_MS`; the dominant fixed cost, trade against cutting callers off |
| STT (short utterance) | 150–400 ms | `ledger_voice_stt_seconds`; Groq / GPU faster-whisper at the low end |
| Retrieval | 20–150 ms | voice caps `top_k` at 3 and bypasses reranking |
| LLM answer | 300–900 ms | full (not token-streamed) answer: citations are validated before anything is spoken |
| TTS first byte | 100–300 ms | `ledger_voice_tts_first_byte_seconds`; response body is streamed, first chunk forwarded immediately |
| **End-of-utterance → first audio** | **≈1.0–1.7 s** (+ VAD hangover ≈ perceived 1.5–2.2 s) | `ledger_voice_first_audio_seconds` |

**Barge-in.** While SPEAKING (until Twilio echoes the turn's `mark`, i.e. until playback actually finished), caller speech lasting `LEDGER_VAD_BARGE_IN_MS` cancels the TTS task, sends Twilio `clear` (drops its playback buffer) and returns to LISTENING. Speech during TRANSCRIBING/THINKING is treated as a premature endpoint: the turn is cancelled and its audio is prepended to the next utterance.

**Security and bounds.** No media is processed before authentication. `/v1/voice/stream` requires `Authorization: Bearer` on the handshake. Twilio cannot set headers on its stream, so `/v1/voice/twilio` also accepts the token as `<Parameter name="token">` (`start.customParameters.token`); in production this must be a short-lived, single-call token minted by the TwiML webhook after validating `X-Twilio-Signature` — never a long-lived user credential. Requests carrying an `Origin` header (browsers) are rejected to prevent cross-site WebSocket hijacking; the `start` event must arrive within 10 s. Both endpoints require the `reader` role and the rate limiter; sessions are capped by `LEDGER_MAX_VOICE_SESSIONS` (1013 when full), session duration, frame size (1009), utterance length, per-turn agent audio and answer length, and each RAG turn still takes a shared query slot. Sends to slow clients time out; disconnects cancel in-flight STT/RAG/TTS tasks. Transcripts and answers are never logged — only lengths and timings. STT/TTS errors are sanitized `SpeechError`s; clients never follow redirects; production requires HTTPS speech endpoints.

**Limitations.** Energy VAD is sensitive to line noise (tune `LEDGER_VAD_SPEECH_RMS` or swap in a neural VAD behind `EnergyVad.process`); resampling is linear with a boxcar anti-alias filter (telephony grade); voice concurrency and rate limits are per process.
