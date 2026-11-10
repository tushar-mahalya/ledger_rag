# ADR 0008: Program-of-thought reasoning core and benchmark-comparable evaluation

- Status: accepted; implemented and measured 2026-11-09 (8-model dev matrix,
  frozen-config FinQA and TAT-QA test runs, retrieved-evidence FinQA). Results:
  [Evaluation](../EVALUATION.md#raw-model-vs-ledger-backend-program-of-thought-reasoning-2026-11-09)
- Date: 2026-11-09

## Context

The goal is a financial RAG backend whose answers are evidence-based and that
reaches or beats published state of the art on benchmarks people trust, **with
small open models**. The 2026-11-01 baseline read gold FinQA pages with
Qwen3.8-27B and a single free-text answer: 72.5% (heuristic score) on dev with the
gold page and 58.9% with retrieved context. Three problems made that system both
weaker and incomparable with the literature:

1. **The model did arithmetic and transcription.** Free-text answers mix operand
   choice, copying, scale and arithmetic errors; small models fail most of these.
2. **Scores were not the official metric.** FinQA reports *execution accuracy*:
   the official executor runs a program and its result must equal `exe_ans`.
   A heuristic numeric match cannot be placed next to published numbers.
3. **Settings were mixed.** Published FinQA/ConvFinQA/TAT-QA numbers give the
   gold page; open-corpus retrieval is a different, harder task (DocFinQA,
   T2-RAGBench, FinanceBench). Each must be reported in its own setting.

## Evidence reviewed (2026-11-09)

| Benchmark (metric, split) | Best peer-reviewed result found | Notes |
|---|---|---|
| FinQA (execution acc., test 1,147) | 75.32 FINDER, GPT-4 PoT + dynamic exemplars ([Khatuya et al., 2025](https://arxiv.org/abs/2510.13157)); APOLLO 71.07; ENCORE 69.40; PoT-GPT-4 69.38 (reproduced); FinQANet 61.24 | Human experts 91.16. Preprints claim ~76.8 (TAT-LLM 70B, an agentic RAG); Fin-R1 7B reports 76.0 on its own evaluation. |
| ConvFinQA (execution acc., dev) | 81.95 FINDER; APOLLO 78.76 | Test answers are private. |
| TAT-QA (EM / F1, test) | 81.4 / 88.4 TAT-LLM 70B ([Zhu et al., 2024](https://arxiv.org/abs/2401.13223)); TAT-LLM 7B 82.9 F1; GPT-4 zero-shot 79.7 F1 | Human 84.1 / 90.8. Test gold is public. |
| FinanceBench (150 open questions) | 19% GPT-4-Turbo with a shared vector store in the original paper; unreviewed agentic systems claim 76–86% | Judged answers; not directly comparable across reports. |
| T2-RAGBench (FinQA+ConvFinQA+TAT-DQA as open-corpus RAG, 2026) | Hybrid + Cohere rerank R@5 .816; end-to-end number match ~.35 | [2604.01733](https://arxiv.org/abs/2604.01733): table-structure mismatch causes 73% of retrieval failures; HyDE hurts numeric QA; reranking is the largest single gain. |

Technique evidence that shaped the design:

- **Program-of-thought beats chain-of-thought for numbers**, with far fewer
  tokens, especially for non-reasoning models ([PoT](https://arxiv.org/abs/2211.12588);
  [FinanceReasoning, ACL 2025](https://arxiv.org/abs/2506.05828): Llama 70B
  71.4% PoT vs 50.4% CoT on the hard set; a reasoner + programmer split fixes
  91.7% of calculation errors).
- **Dynamic in-context exemplars** are the main lever of the current FinQA SOTA
  (FINDER: +5.98 points over APOLLO).
- **Self-consistency** (majority vote over sampled programs) is a standard
  test-time scaling method for table QA; execution-based voting needs no verifier.
- **Small models specialize well**: TAT-LLM 7B beats GPT-4 on TAT-QA after SFT on
  gold derivations; Fin-R1 (7B, SFT + GRPO) reports 76.0 FinQA / 85.0 ConvFinQA.
- **Labels are noisy**: FinanceReasoning corrected or disambiguated 14.2% of its
  FinQA-derived test items, so FinQA accuracy is capped well below 100%.

## Decision

1. **The reader never computes.** `ledger_rag.reasoning` makes the model write a
   FinQA-DSL program (`subtract(5829, 5735), divide(#0, 5735)`); it is parsed and
   bounded (`dsl`), every literal must appear in the evidence or question
   (`grounding`), it is executed deterministically, and N samples vote on the
   executed value (`consistency`). Similar *training* problems with gold programs
   are retrieved by BM25 as exemplars (`exemplars`). The DSL is not Python, so no
   sandbox is needed and tools stay bounded and read-only.
2. **Official metrics only for benchmark claims.** The FinQA executor is vendored
   byte-for-byte (MIT, hash-pinned by a test); it reproduces all 8,281 FinQA gold
   `exe_ans` values and, with the documented bare-number lookup rule, ConvFinQA's.
   Reports give the official exact score and, separately, the PoT tolerant score
   used by LLM-prompting papers, with report-cluster bootstrap intervals.
3. **Settings are explicit.** `finqa_reasoning` measures the standard gold-page
   setting. Retrieval quality and open-corpus answering remain in
   `finqa`/`finqa_answers` and are reported separately.
4. **Test splits stay sealed** until a configuration is frozen on dev
   (`--unseal-test`); the Modal job cannot reach the test split at all.
5. **Compute is colocated and bounded.** `compute/reasoning.py` runs vLLM on
   loopback and the benchmark in one GPU container with a hard timeout; nothing is
   deployed. Weights and compile caches persist in Volumes.
6. **The backend, not the model, is what is measured.** Every run compares arms
   of the *same* model on identical questions and generation settings:
   `raw` (the model alone: same evidence, free reasoning, one `ANSWER:`) versus
   `ledger` (exemplars, programs or typed answers, grounding, deterministic
   execution, voting), plus ablations such as `ledger-zero-shot`. Reports carry
   paired report-cluster bootstrap deltas (`ledger-minus-raw`). Raw answers are
   parsed leniently and also scored with the PoT percentage-tolerant comparison,
   so the control is never penalized for formatting; gold replay checks that the
   raw parser scores 99.7–100% on gold answers.
7. **Any LLM, one interface.** Models are declared in `config/models/<id>.toml`
   (`ledger_rag.evaluation.models`): pinned self-hosted weights served by vLLM, or
   a hosted OpenAI-compatible API, each with named generation modes. Adding a
   model is a file, not code; the Modal job runs many models in parallel, one
   container per model serving all benchmarks and arms.

## Budget (authorized 2026-11-09: $25 of Modal credit)

| Phase | Allocation | Gate to proceed |
|---|---:|---|
| Multi-model raw-vs-Ledger dev matrices (8 models × 3 benchmarks) | ~$9 | Pipeline bugs fixed on 50/100-question pilots first |
| Open-corpus (retrieved evidence) FinQA and ablations | ~$3 | Gold-evidence results analyzed first |
| Frozen held-out test runs (FinQA test, TAT-QA test) | ~$3 | Configuration frozen on dev |
| Contingency, then optional small-model SFT | ~$10 | Backend gains measured first; SFT changes the model, not the backend |

Reallocated after the user asked that the backend, not a tuned model, carry the
result: comparing many models raw versus Ledger now takes precedence over
fine-tuning one. Spend is tracked per run (GPU seconds x list rates) in
[Evaluation](../EVALUATION.md#raw-model-vs-ledger-backend-program-of-thought-reasoning-2026-11-09).

Every run records GPU, timings and its report under `ledger-data:/reasoning/`.
List rates (2026-11-09): H100 $3.95/h, L40S $1.95/h, CPU $0.047/core-h,
memory $0.008/GiB-h.

## Roadmap (ordered by expected gain per dollar)

1. Measure the reasoning core on FinQA/ConvFinQA dev with 4B/9B/27B readers;
   tune only on dev (exemplar count, samples, thinking mode).
2. TAT-QA: vendor the official metric (needs NumPy/SciPy), add span/count answers
   and scale prediction to the reader contract.
3. Specialize Qwen3.5-4B: SFT on FinQA + ConvFinQA + TAT-QA train programs, then,
   if budget allows, GRPO with an execution-accuracy reward.
4. Open-corpus RAG: table-cell-aware chunks and retrieval (the dominant failure
   in T2-RAGBench), then end-to-end FinQA/T2-RAGBench/DocFinQA with retrieved
   evidence and the same reader.
5. Production: route chat numeric questions through the solver (sampled programs,
   evidence-bound operands already enforced by `generation.program`); voice keeps
   one greedy sample within its latency budget.

## Rejected or deferred

- **Python program execution**: more expressive, but needs a sandbox and admits
  arbitrary code; the DSL covers every FinQA/ConvFinQA gold program.
- **Free-text answers with LLM judges** for FinQA-family claims: incomparable
  with published execution accuracy.
- **Bigger proprietary readers**: they would raise the score but defeat the goal of
  showing that the pipeline, not the model size, carries the result.
- **Tuning on test**: never; test is reporting-only.
