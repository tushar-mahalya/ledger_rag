# Evaluation: evidence, uncertainty, and operating envelopes

This repository ships **synthetic regression data**, not a financial benchmark
leaderboard or evidence of state-of-the-art performance. The local baseline is a
deterministic lexical retriever and extractive provider. It exists to exercise
contracts without keys, network traffic, model downloads, or benchmark redistribution.
Do not relabel its scores as FinanceBench, FinQA, or TAT-QA results.

## 1. Executable evaluation contract

Run from the repository root:

```sh
uv run python -m ledger_rag.evaluation.harness --smoke
uv run python -m ledger_rag.evaluation.harness \
  --corpus data/examples.jsonl --cases data/eval.jsonl \
  --repeats 3 --top-k 5 --seed 42 --bootstrap-samples 2000 \
  --output evaluation.json
uv run python -m pytest tests/test_evaluation.py -q
```

The default CLI reports measurements; it does **not** demand perfection.
`--smoke` explicitly gates the bundled synthetic labels at document recall,
abstention correctness, and labeled answer-substring accuracy of 1. These are
regression invariants, not realistic deployment targets. `--thresholds PATH`
replaces smoke gates with a preregistered JSON policy. Failed gates exit 1;
configuration/data/runtime failure exits 2 without printing exception contents
that could include credentials. A run fails fast on runtime errors rather than
silently censoring failed queries and averaging only successes. Evaluation is
sequential, and each retrieval/generation pair has the configured query deadline.
Latency begins after corpus loading/indexing; indexing, bootstrap, and report
serialization are not included.

### Case format (JSONL)

Required: unique `id`, nonempty `question`; answerable cases require
`relevant_document_ids`. Optional:

| Field | Contract |
|---|---|
| `as_of` | ISO date; enforces the service's publication cutoff |
| `relevant_chunk_ids` | Stable positive chunk IDs; omit if not annotated |
| `should_abstain` | Default false; true means insufficient authorized evidence |
| `answer_contains` | Nonempty strings, all required case-insensitive matches |
| `numerical_assertions` | Gold numbers, explicit tolerance, optional contextual regex |
| `slices` | Multiple labels, e.g. numerical, multi-document, temporal, ACL |
| `split` | Train/dev/test label used in duplicate-question leakage reporting |

Unknown label fields, empty substring assertions, unknown document/chunk IDs,
duplicate case IDs, duplicate corpus document IDs within a tenant, empty corpora,
and empty case sets are rejected. An empty chunk-positive list on an answerable
case is rejected: absence of annotation is **not** a negative judgment. Gold labels
can describe documents that become inaccessible under the chosen principal; a
deliberate negative ACL case should set `should_abstain=true`.

Example using **invented** content, not benchmark material:

```json
{
  "id": "synthetic-numeric",
  "question": "What was ExampleCo revenue?",
  "relevant_document_ids": ["example-report"],
  "numerical_assertions": [{
    "expected": 120,
    "absolute_tolerance": 0.01,
    "relative_tolerance": 0,
    "pattern": "Revenue was ([0-9.,]+) million"
  }],
  "slices": ["numerical", "single-document"],
  "split": "test"
}
```

Numerical assertions accept a finite matched value `x` when
`abs(x-gold) <= max(absolute_tolerance, relative_tolerance * max(abs(x), abs(gold)))`,
following `math.isclose`. A pattern may capture at most one group. The default
number matcher supports signs, commas, decimals, and scientific notation; it
does **not** interpret units, percentage scaling, accounting parentheses,
currencies, or the roles of different numbers. Use a contextual pattern with
gold-specified units; any matching number can otherwise satisfy the assertion.
All numerical assertions must pass. A matching number somewhere in an answer
does not prove the arithmetic, argument, or cited inputs are correct.
Patterns and cases are trusted evaluation inputs, not a public regex execution API.

## 2. Metrics and denominators

For a ranked list, remove duplicate identities **before** taking the first `k`.
Document identities are document IDs; chunk identities are chunk IDs. Multiple
chunks from the same document do not create multiple document true positives or
push down document ranks. Chunk evaluation still treats genuinely distinct
chunks as distinct evidence. Different IDs containing the same text require
dataset-side deduplication: the evaluator does not invent equivalence judgments.

For positive gold set `G` and unique ranked prefix `R_k`:

* **Recall@k:** `|R_k ∩ G| / |G|`.
* **Precision@k:** `|R_k ∩ G| / k`, including missing results as empty positions.
  Returning one correct document at `k=5` gives precision 0.2, not 1.
* **MRR@k:** reciprocal rank of the first positive, or zero when none appears.
* **nDCG@k:** binary-relevance `sum(1/log2(rank+1))` over positives divided by
  the ideal sum over `min(k, |G|)` positions. The ideal denominator does not
  shrink when the retriever returns too few items. No graded relevance is implied.

`document_*_at_k` and `chunk_*_at_k` are separately reported. An omitted
chunk annotation yields null chunk metrics. Empty positive gold is undefined
(`null`), never a perfect recall or nDCG score. Cases marked `should_abstain`
are excluded from **all positive retrieval aggregates**, even if they contain
document labels. Their retrieval rows remain inspectable.
Unjudged retrieved IDs count as nonrelevant under a closed-world gold assumption;
incomplete labels will underestimate relevance.

The cutoff is the requested `--top-k` for chat and `min(top_k, 3)` for voice,
recorded as `effective_k`. The service candidate budget counts chunks;
deduplication can leave fewer than `k` unique documents. The evaluator does
not secretly overfetch to fill missing document slots.
Retrieval metrics measure **initial retrieval**.
The chat corrective search tool may later retrieve additional evidence;
final citation-document precision measures the final cited document set instead.
Do not claim initial-retrieval recall measures the entire tool trajectory.

### Abstention and selective prediction

The positive class is **should abstain**:

| Gold / prediction | Abstains | Answers |
|---|---:|---:|
| Should abstain | TP | FN: unsupported answer |
| Answerable | FP: needless refusal | TN |

Sensitivity is `TP/(TP+FN)`; specificity is `TN/(TN+FP)`. Missing classes
yield null, not perfect sensitivity/specificity. Counts are run counts, including
repeats. `abstention_correct` reports both classes but can conceal imbalance,
so inspect the confusion table and each class-specific rate.

Coverage is the fraction of runs answering. Risk on labeled answered runs is
the fraction failing supplied answer assertions, including an answer on a
should-abstain case as a failure. Answered positive cases with no substring or
numeric labels have **unknown answer correctness**, not automatic success.
Report the labeled and unlabeled answered denominators beside risk.
All-abstain systems have zero coverage and undefined answered risk, not zero
risk. Do not optimize abstention correctness without minimum coverage and
answerable-specificity gates.

### Citation and answer checks: deliberately limited claims

`citation_document_precision` is the fraction of **unique cited documents**
in the gold document set. Duplicate citation markers/documents do not change it.
No citations yields null, not a reward for abstention. A cited wrong document
counts as nonrelevant, including on an unanswerable case with empty gold.
Its aggregate uses only runs with citations; consult that metric's eligible-case
count in the confidence interval report.

**Citation-document precision is NOT semantic faithfulness.** Even a relevant
document may not support a particular claim, quote, date, denominator, or unit.
Substring/numeric scores measure only explicit assertions. Missing assertions
are null, not vacuous passing tests. An asserted string can appear in a false,
negated, or contradictory answer. None of these metrics constitutes a semantic
judge, financial correctness certification, or grounded-answer accuracy.

## 3. Uncertainty, reproducibility, and leakage

Each metric first averages repeated runs within a case, then takes a macro
mean across eligible cases. The **seeded nonparametric percentile bootstrap**
resamples these case means with replacement and reports a 95% interval
(nearest-rank 2.5th/97.5th percentiles). Repeats are not independent questions;
they do not inflate bootstrap sample size. With a single case or identical
scores the interval degenerates; empty samples yield null.
Intervals describe this case sample, not uncertainty over all possible filings,
companies, production traffic, or model pretraining.

`--seed` seeds the bootstrap only. It does not control a remote model's sampler,
network scheduling, or indexing randomness. Repeated normalized questions under
different IDs are flagged, but are not silently removed or reweighted. Where
questions share reports/companies, case-level independence is optimistic:
publication-quality studies should also bootstrap at company/report level,
use paired case-level differences for model comparisons, and state cluster units.
Those grouped/paired comparison procedures are experiment extensions, not
implemented claims of this CLI.

The manifest records exact input-byte SHA256 digests, installed package and
Python versions, package Python-source SHA256, model/provider identifiers, and an
allowlist of evaluation/model/retrieval configuration. It never serializes
`Settings.model_dump()`, tokens, keys, endpoint URLs, or exception bodies.
Model identifiers may be mutable aliases: the manifest explicitly marks model
revision as unreported. For audit-quality experiments independently attach
immutable provider revisions/checkpoint hashes, prompt fingerprints, extraction
and chunking revision, dependency lockfile digest, index snapshot/digest,
hardware/region, tokenizer, generation sampling parameters, budget/cost,
and dataset license/version ledger. These are requirements for the experiment
record, not fabricated runtime metadata.

The leakage report casefolds Unicode and collapses whitespace, lists duplicate
case IDs/splits using a question SHA256, and flags cross-split groups.
Repeats are excluded. It detects **exact normalized questions only**, not
semantic paraphrases, shared tables, duplicate disclosures, or pretraining
contamination. Split by issuer/report/time before question generation; audit
near duplicates and template families separately. Keep answer, rationale,
program, gold evidence labels, and expected outputs out of indexed documents.
Public benchmark test sets are potentially present in LLM pretraining. Report
that limitation and evaluate separately on a rights-cleared temporal holdout.

## 4. Remote evaluation and authorization boundaries

```sh
# Configure deployment settings/credentials through environment or secret files first.
uv run python -m ledger_rag.evaluation.harness --remote \
  --tenant YOUR_AUTHORIZED_TEST_TENANT --groups YOUR_AUTHORIZED_TEST_GROUP \
  --allow-index-write --corpus your-corpus.jsonl --cases your-cases.jsonl \
  --repeats 3 --output remote-evaluation.json
```

Remote uses the same environment/secret-file settings loader and retriever factory
as the service, including configured Qdrant/hybrid retrieval and optional TEI
reranking. It can incur embedding, generation, and reranking costs.
The local baseline explicitly overrides ambient remote/production/reranker
settings; it stays memory/extractive.

**This evaluator calls the service directly, not through HTTP authentication.**
`--tenant` is mandatory remotely; group IDs are explicit, with no anonymous or
all-access fallback. These are operator-supplied authorization test scopes, not
proof of JWT authentication or permission to impersonate a production identity.
Use only authorized test principals and a disposable isolated index.

Qdrant evaluation **upserts the corpus**, possibly replacing existing tenant/
document entries and creating embeddings. It requires `--allow-index-write`
before any index client is constructed. That flag is not an ACL bypass or a
license to alter a production collection. Other/stale documents can remain in
the collection, so a corpus hash alone is not an index snapshot; isolate and
version the collection. Offline ACL filtering checks do not certify HTTP auth,
OIDC JWKS handling, ingestion authorization, tenant-bound writes, or cache safety.

## 5. Preregister experiments and slice gates

Use identical corpora, chunk boundaries, gold judgments, as-of dates, principals,
profiles, and generation settings across configurations. Freeze a dev-selected
policy before running the test set:

| Experiment | What changes | What must be held/measured |
|---|---|---|
| Sparse baseline | Local lexical scoring | Tokenization, k, gold scope; not a production sparse-model result |
| Dense-only | External embedding retrieval without sparse fusion | Embedding revision, distance, index settings, candidate budget |
| Hybrid | Dense + sparse candidates/fusion | Same dense model, sparse model, fusion rule, filtered candidates |
| Hybrid + rerank | Cross-encoder reranker | Reranker revision, candidate count, timeout, final k, compute/cost |
| Corrective chat tools | Search/calculator rounds | Initial vs final evidence, tool budget, extra latency and failure rate |

Dense-only/alternative sparse ablations require an independently configured
retriever/index experiment; the CLI has no fictitious `--dense-only` switch.
The configured production factory determines the retrieval implementation.
The TEI reranker considers up to 10 candidates for chat and voice bypasses it;
report this asymmetry rather than calling chat and voice latency comparable.

Example **illustrative**, not certified, thresholds:

```json
{
  "minimum": {
    "document_recall_at_k": 0.80,
    "abstention_sensitivity": 0.90,
    "abstention_specificity": 0.85,
    "coverage": 0.60
  },
  "maximum": {"risk_on_labeled_answers": 0.10},
  "maximum_latency_ms": {"total_p95": 3000},
  "slices": {
    "numerical": {"minimum": {"numerical_assertions_pass": 0.90}},
    "acl-negative": {"minimum": {"abstention_sensitivity": 1.0}}
  }
}
```

Unknown metric keys and nonfinite bounds are rejected. A requested metric with
no eligible labels, or a missing slice, fails the gate. Gates use point estimates;
requiring lower-confidence-bound guarantees is an additional policy, not an
implemented guarantee. Averages can hide critical failures: require sufficient
slice sizes and inspect numerical, table+text, multi-document, ambiguous-period,
outdated/future disclosure, unanswerable, adversarial, ACL-positive/negative,
tenant collision, and voice/chat slices. Slice membership may overlap.

**Additional gold data/judges are necessary** for semantic faithfulness and
claim-level citation entailment, numeric consistency with units and derivations,
credible unanswerability labels, adversarial instruction resistance, and
production-grade ACL suites. Use blinded human financial-domain review with
rubric, adjudication, and inter-rater agreement; if using a model judge, pin it,
calibrate against human labels, disclose disagreement and judge cost, and avoid
same-model self-judging. Never report those suites as passing from substring
checks or fabricate SOTA comparisons. CI smoke verifies invariants only.

## 6. Financial benchmark adapters and extension design (no bundled data)

Adapters must be **offline, opt-in converters** for data acquired with
appropriate rights. They must write the corpus/Case JSONL contract, plus a
provenance sidecar of source revision, original example IDs, split, source-file
hashes, extraction/chunk mapping, table serialization rules, license/attribution,
and conversion version. Do not package benchmark questions, answers, filing
passages, or source PDFs in this repository. Dataset annotation licenses do
not automatically clear underlying issuer-authored disclosure copyrights.
Public accessibility on SEC EDGAR does not make privately authored filings
US-government-authored public-domain works.

### FinanceBench

Primary sources: [maintainer repository](https://github.com/patronus-ai/financebench),
[paper](https://arxiv.org/abs/2311.11944),
[maintainer dataset card](https://huggingface.co/datasets/PatronusAI/financebench).

The public repository describes an **open sample of 150 annotated examples**;
the full 10,231-question dataset is not the public sample. The maintainer dataset
card declares **CC BY-NC 4.0** for the released dataset: attribution and
non-commercial restrictions matter. Source-document rights need separate review.
Do not imply possession of the full benchmark or commercial clearance.

**Implemented retrieval-only preparation:**

```sh
uv run python scripts/prepare_financebench.py \
  --input licensed-financebench.jsonl --corpus independently-prepared-filings.jsonl \
  --mapping financebench-document-mapping.json --output financebench-cases-v1.jsonl \
  --accept-dataset-terms
uv run python -m ledger_rag.evaluation.harness \
  --corpus independently-prepared-filings.jsonl --cases financebench-cases-v1.jsonl
```

This converter does not download data. It requires user-provided licensed
FinanceBench JSONL and an **independently prepared full-filing corpus**, not
gold evidence snippets. The mapping JSON contains
`{"document_ids":{"FINANCEBENCH_DOC_NAME":"CORPUS_DOCUMENT_ID"},"corpus_sha256":"64hex"}`;
the digest must match the exact corpus bytes. Output paths must be new, allowing
versioned cases and a `.manifest.json` sidecar. That sidecar records adapter
version and input/corpus/mapping SHA256 digests.

It maps `financebench_id`, `question`, and `doc_name` to
`Case{id,question,relevant_document_ids,should_abstain:false}`. It deliberately
does **not** copy gold answers, evidence, rationale, chunk labels, numerical
assertions, or publication dates into cases/corpus. Evaluation still invokes
the provider, but answer-assertion metrics remain null; document relevance
and citation-document precision are not semantic-answer scores. No unanswerable
cases means abstention sensitivity is undefined. The converter uses primary
`doc_name` relevance, not an audited union of all evidence documents. Its valid
claim is **document retrieval on the converted sample**, not complete
FinanceBench QA performance, page localization, or official benchmark scoring.
The accept-terms flag records explicit opt-in; it does not establish legal rights.

**A richer adapter remains extension design, not current implementation:**
map `financebench_id` to namespaced case ID, `question` to query, and each
`evidence_doc_name` to positive document ID. Join `doc_name` metadata to original
document links. Map zero-indexed `evidence_page_num` plus evidence spans to
generated chunk IDs with an audited page-offset map. Obtain publication dates
from filing metadata, not fiscal-year guesses. Keep annotated answers,
justifications, question types/reasoning labels, and evidence annotations in
gold sidecars, not in the corpus. Document-level positives alone are too weak
for page/span faithfulness. Use human/validated semantic assessment for free-form
answers; numerical assertions only where units and the target field are explicit.

### FinQA

Primary sources: [maintainer repository](https://github.com/czyssrs/FinQA),
[paper](https://arxiv.org/abs/2109.00122),
[repository license](https://github.com/czyssrs/FinQA/blob/main/LICENSE),
[IBM rehost card](https://huggingface.co/datasets/ibm-research/finqa).

The GitHub repository license is **MIT**, while the IBM dataset rehost card
declares **CC BY 4.0**. Record the artifact/revision acquired and resolve this
discrepancy for intended use; do not silently choose the most permissive label.
Neither claim automatically clears underlying source report rights.

Preserve native train/dev/test boundaries. Reconstruct report/page identity from
example IDs; serialize `pre_text`, `table`, and `post_text` consistently for both
positive and negative candidates, preserving row/column headers and units.
Keep `qa.program`, `qa.exe_ans`, and `qa.gold_inds` exclusively in gold labels.
Map supporting fact indices to audited chunks. Gold programs/execution results
can support an independent official execution/program scorer; the current
substring/numeric evaluator is **not** that scorer. The private test has no
public reference labels; do not invent retrieval or execution accuracy there.

The maintainer changelog documents a May 2022 `table_row_to_text` formatting bug
that leaked labels and inflated results. Pin a corrected revision and test
positive/negative serialization parity. That construction leak is separate from
LLM-pretraining contamination; both require disclosure.

**Local Qdrant retrieval/quantization experiment (optional):**

```sh
# One complete, idempotent upsert into a dedicated local collection.
uv run --locked python -m ledger_rag.evaluation.benchmarks.finqa ingest \
  --data data/finqa --qdrant-url http://127.0.0.1:16333 \
  --collection finqa_bench --embedding-url http://127.0.0.1:18080/v1 \
  --embedding-key local --embedding-model Alibaba-NLP/gte-modernbert-base \
  --embedding-dimensions 768 --output benchmarks/results/finqa-ingest.json

# Copy stored vectors without another embedding pass; each destination must be new.
uv run --locked python scripts/clone_qdrant_benchmark.py \
  --qdrant-url http://127.0.0.1:16333 --source finqa_bench \
  --destination finqa_bench_scalar --quantization scalar
uv run --locked python scripts/clone_qdrant_benchmark.py \
  --qdrant-url http://127.0.0.1:16333 --source finqa_bench \
  --destination finqa_bench_binary --quantization binary

# Run the same full test split, scopes and retrieval arms for each index.
uv run --locked python -m ledger_rag.evaluation.benchmarks.finqa retrieval \
  --data data/finqa --qdrant-url http://127.0.0.1:16333 \
  --collection finqa_bench --quantization none --oversampling 2 \
  --embedding-url http://127.0.0.1:18080/v1 --embedding-key local \
  --embedding-model Alibaba-NLP/gte-modernbert-base --embedding-dimensions 768 \
  --reranker-url http://127.0.0.1:18081 \
  --configs sparse dense hybrid hybrid:rerank --scopes company global \
  --output benchmarks/results/finqa-retrieval-none.json
```

Repeat the retrieval command with destination/quantization pairs
`finqa_bench_scalar`/`scalar` and `finqa_bench_binary`/`binary`. Keep the test
split, model, `--configs`, scopes, top-k and hardware identical; record the
ingest/copy manifests and actual invocation. Collections are separate so search
never mutates the full-precision baseline. Vector copying preserves payloads and
original vectors; the helper refuses an existing destination, verifies the source
schema and exact point counts, and never deletes either collection.

The CLI default evaluates the full 1,147 test questions. `--limit N` selects the
first N rows (not a randomized representative sample) and is suitable only as a
smoke/pilot. Report it as such. `--quantization none` explicitly bypasses any
stored compressed index; scalar/binary settings must match their effective dense
vector configuration. Report Recall@k/evidence recall, p50/p95/p99, sample count,
oversampling, point count and RAM/disk. No accuracy or latency gain is assumed;
FinQA split/license and source-filing rights caveats above still apply.

### TAT-QA

Primary sources: [maintainer repository](https://github.com/NExTplusplus/TAT-QA),
[project site](https://nextplusplus.github.io/TAT-QA/),
[ACL paper](https://aclanthology.org/2021.acl-long.254/).

The repository states **CC BY 4.0** for the dataset (attribution required), with
16,552 questions across 2,757 hybrid table/text contexts. It announced release
of test ground truth in January 2024: pin the exact revision and do not mix
earlier hidden-label and later public-label evaluations.

Map context/table/paragraph IDs to stable source/chunk identities; preserve cell
coordinates, headers, paragraph order, arithmetic derivations, answer types and
scales in conversion sidecars. Multi-span answers need span/set scoring; numeric
answers need explicit percentage/million/billion normalization. The official
normalized exact-match/F1 methodology should run separately from this evaluator.
Local document recall is not automatically comparable to supplied-context QA.
Keep gold answers/derivations outside indexed content; group splits by shared
report/context to prevent multiple questions on one table crossing folds.

## 7. HTTP load experiments: event time is not answer time

The harness uses authenticated HTTP only; there are no tenant/group impersonation
headers or anonymous fallback. Configure a valid reader bearer token through the
environment named by `--token-env` (default `LEDGER_BENCHMARK_TOKEN`), never as a
CLI token argument or URL credential. Do not print tokens in shell history.

```sh
uv run python scripts/benchmark_http.py --requests 200 --concurrency 4 \
  --warmup 10 --profile chat --output load-chat.json
uv run python scripts/benchmark_http.py --duration 30 --concurrency 4 \
  --warmup 10 --profile voice --stream --output load-voice.json
```

Budgets: exactly one request count or arrival duration (default 100 requests),
concurrency 1–1000, requests up to 1,000,000, duration up to 3600s, warmup up to
10,000, and a finite absolute per-request timeout up to 180s. A fixed worker pool
keeps in-flight queries bounded. Duration stops new arrivals; already-started
requests drain within their timeout. Warmup is sequential and excluded from
measured latency/throughput; warmup failures are reported separately.
The response budget is 2 MiB. Repeated questions exercise cache-friendly traffic;
they are not a representative query distribution.

The JSON report separates:

* Attempted/successful requests, success rate, errors by sanitized category,
  throughput (attempts/elapsed wall time), and goodput (valid answers/elapsed).
* Nearest-rank p50/p95/p99 all-request completion (including failures).
* Successful-request completion and validated-answer latency, with sample counts.
* **SSE TTFE:** receipt of a complete, parsed `evidence` event. It can contain
  zero sources; TTFE is not evidence relevance, first token, or time to speech.
* **Validated-answer latency:** receipt and basic wire-contract validation of
  the complete `answer` event, or complete JSON answer for nonstreaming requests.
  This checks profile, answer/citation presence, and finite server timings,
  not semantic truth. Server-side citation validation is a separate guarantee.

SSE success requires ordered evidence/answer/done events and complete termination.
An HTTP 200 stream that later errors, truncates, duplicates terminal events, or
times out is **not** success, even if an answer event was observed.
Successful TTFE/answer percentiles exclude these failures; error counts and
all-request completion expose that survivorship limitation. Tokens, endpoint
URLs, response bodies, and exception content are never included in the report.
Redirects are not followed; environment proxies are disabled to avoid unintended
credential forwarding. A default success-rate gate of 0.99 exits 1 on failure;
`--min-success-rate` sets the preregistered alternative.

This is a **closed-loop** benchmark, not an open-loop arrival generator.
It can hide coordinated omission: stalled workers reduce arrivals.
Sweep concurrency and separate chat/voice, warm/cold, model/index revisions,
same-region/cross-region, and authorized/unanswerable workloads. Record rate
limits, quotas, resource saturation, errors, and cost beside latency. At least
100 successful observations are needed before p99 even addresses a distinct
upper percentile position; far more are needed for a stable tail estimate.
No performance numbers or production capacity claims are asserted by this doc.
