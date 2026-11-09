"""Program-of-thought benchmarks in the standard gold-evidence setting.

Every question is answered from its own evidence (a FinQA/ConvFinQA report page, a
TAT-QA table with paragraphs), as in each benchmark's paper and leaderboard, so the
scores are comparable with published results. Open-corpus retrieval is measured
separately (:mod:`ledger_rag.evaluation.benchmarks.finqa`); never mix the settings.

Pipeline per question: dynamic exemplars from the training split -> N sampled
replies -> parse, ground, execute -> majority vote on the executed outcome.

| Benchmark | Contract | Metrics (official first) |
|---|---|---|
| ``finqa`` | DSL program | execution accuracy (official ``evaluate.py``), PoT tolerant |
| ``convfinqa`` | DSL program per turn, gold history | same as FinQA |
| ``tatqa`` | typed answer + scale | official TAT-QA exact match and F1 |

Readers: ``replay-gold`` answers with the gold solution (an offline pipeline check
that spends nothing); ``endpoint`` calls an OpenAI-compatible chat endpoint.
Test splits are sealed: ``--split test`` needs ``--unseal-test`` and a configuration
frozen on dev. ConvFinQA test answers are private.

Usage::

    python -m ledger_rag.evaluation.benchmarks.reasoning --benchmark tatqa --reader replay-gold
    python -m ledger_rag.evaluation.benchmarks.reasoning --reader endpoint \\
        --llm-model Qwen/Qwen3.5-4B --samples 5 --exemplars 4 --output run.json
"""

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ledger_rag.evaluation import models
from ledger_rag.evaluation.datasets.fetch import verify
from ledger_rag.evaluation.llm import ChatClient, GenerationConfig
from ledger_rag.evaluation.official import finqa as official
from ledger_rag.evaluation.uncertainty import (
    latency,
    paired_cluster_difference,
    risk_coverage,
    summarize_rows,
)
from ledger_rag.reasoning import direct, typed
from ledger_rag.reasoning.consistency import Candidate, decide
from ledger_rag.reasoning.dsl import Program, ProgramParseError
from ledger_rag.reasoning.dsl import parse as parse_dsl
from ledger_rag.reasoning.exemplars import Exemplar, ExemplarIndex
from ledger_rag.reasoning.expression import render as render_expression
from ledger_rag.reasoning.solver import (
    PAGE_MARKER,
    PROMPT_VERSION,
    Execution,
    Problem,
    ProgramTask,
    Reply,
    Solution,
    Solver,
    Task,
    problem_text,
)

SPLITS = ("train", "dev", "test")
READERS = ("replay-gold", "endpoint")
CONVFINQA_FILES = {"train": "data/train.json", "dev": "data/dev.json"}
TATQA_FILES = {
    "train": "tatqa_dataset_train.json",
    "dev": "tatqa_dataset_dev.json",
    "test": "tatqa_dataset_test_gold.json",
}


@dataclasses.dataclass(frozen=True)
class Item:
    """One scored question: the problem, its gold answer and solution, and cluster."""

    problem: Problem
    gold: Any
    solution: str
    """Gold reply lines after FACTS, used by ``replay-gold`` and as exemplar text."""
    cluster: str
    raw_solution: str = ""
    """Gold ``ANSWER:`` lines in the raw contract, for ``replay-gold`` of raw arms."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split(directory: Path, split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = json.loads((directory / f"{split}.json").read_text())
    return records


def report_cluster(filename: str) -> str:
    """Bootstrap cluster: one annual report (``company/year``)."""
    return "/".join(filename.split("/")[:2])


def program_line(dsl: str) -> str:
    """A gold DSL program as the expression line a reader writes."""
    return f"PROGRAM: {render_expression(parse_dsl(dsl))}"


def official_executor(program: Program, item: Problem) -> Execution:
    result = official.execute(program.text, item.table)
    return Execution(result.value, result.valid)


def _page(record: dict[str, Any], **fields: Any) -> Problem:
    return Problem(
        pre_text=list(record["pre_text"]),
        post_text=list(record["post_text"]),
        table=[list(row) for row in record["table"]],
        group=str(record["filename"]),
        **fields,
    )


# --- FinQA ---------------------------------------------------------------------------


def finqa_items(directory: Path, split: str) -> list[Item]:
    return [
        Item(
            _page(record, id=str(record["id"]), question=str(record["qa"]["question"])),
            record["qa"]["exe_ans"],
            program_line(str(record["qa"]["program"])),
            report_cluster(str(record["filename"])),
            f"ANSWER: {record['qa']['exe_ans']}",
        )
        for record in load_split(directory, split)
    ]


def finqa_exemplars(directory: Path) -> list[Exemplar]:
    return [
        Exemplar(
            id=str(record["id"]),
            question=str(record["qa"]["question"]),
            facts=tuple(str(t).strip() for t in record["qa"].get("gold_inds", {}).values()),
            solution=program_line(str(record["qa"]["program"])),
            group=str(record["filename"]),
        )
        for record in load_split(directory, "train")
    ]


def retrieved_contexts(path: Path) -> dict[str, Any]:
    """The ``contexts`` block of a saved retrieval report (``finqa_answers`` contexts)."""
    report: dict[str, Any] = json.loads(path.read_text())
    block: dict[str, Any] = report.get("contexts", report)
    contexts: dict[str, Any] = block.get("contexts", block)
    if "hits" not in contexts:
        raise ValueError(f"{path} holds no retrieval hits")
    contexts["lineage"] = block.get("lineage", {})
    return contexts


def markdown_rows(text: str) -> list[list[str]]:
    """Table rows rendered as Markdown inside a chunk (``| a | b |``), separators dropped."""
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [cell.strip() for cell in stripped[1:-1].split("|")]
        if all(set(cell) <= set("-: ") for cell in cells):
            continue
        rows.append(cells)
    return rows


def finqa_retrieved_items(directory: Path, split: str, path: Path) -> list[Item]:
    """FinQA questions answered from saved top-k retrieved chunks, not the gold page."""
    from ledger_rag.evaluation.benchmarks import finqa as finqa_bench  # Corpus rebuild.

    contexts = retrieved_contexts(path)
    corpus, _ = finqa_bench.load(directory, split)
    expected = contexts["lineage"].get("corpus_sha256")
    if expected and expected != finqa_bench.corpus_sha256(corpus):
        raise ValueError("saved contexts were retrieved from a different FinQA corpus")
    items = []
    for item in finqa_items(directory, split):
        rows = contexts["hits"].get(item.problem.id)
        if rows is None:
            raise ValueError(f"saved contexts do not cover {item.problem.id}")
        passages, table = [], []
        for document_id, chunk_id, _score in rows:
            chunk = corpus.evidence(document_id).by_id.get(chunk_id)
            if chunk is None:
                raise ValueError(f"chunk {chunk_id} is not in the rebuilt corpus")
            passages.append(f"{chunk.title}\n{chunk.text}")
            table.extend(markdown_rows(chunk.text))  # Table functions run on retrieved rows.
        problem = Problem(
            id=item.problem.id,
            question=item.problem.question,
            pre_text=[],
            post_text=[],
            table=table,
            group=item.problem.group,
            passages=tuple(passages),
        )
        items.append(dataclasses.replace(item, problem=problem))
    return items


# --- ConvFinQA -----------------------------------------------------------------------


def _answer_text(value: float | int | str) -> str:
    return value if isinstance(value, str) else f"{value:g}"


def convfinqa_conversations(directory: Path, split: str) -> list[dict[str, Any]]:
    with zipfile.ZipFile(directory / "data.zip") as bundle:
        records: list[dict[str, Any]] = json.loads(bundle.read(CONVFINQA_FILES[split]))
    return records


def convfinqa_items(directory: Path, split: str) -> list[Item]:
    """Every turn, with the teacher-forced gold history (the standard turn-level setup)."""
    items: list[Item] = []
    for record in convfinqa_conversations(directory, split):
        annotation = record["annotation"]
        questions, programs = annotation["dialogue_break"], annotation["turn_program"]
        answers = annotation["exe_ans_list"]
        for turn, question in enumerate(questions):
            history = tuple((str(questions[i]), _answer_text(answers[i])) for i in range(turn))
            problem = _page(
                record, id=f"{record['id']}/turn_{turn}", question=str(question), history=history
            )
            items.append(
                Item(
                    problem,
                    answers[turn],
                    program_line(str(programs[turn])),
                    report_cluster(str(record["filename"])),
                    f"ANSWER: {answers[turn]}",
                )
            )
    return items


def convfinqa_exemplars(directory: Path) -> list[Exemplar]:
    """Train turns shown with their conversation; retrieval matches the whole dialogue."""
    exemplars = []
    for item in convfinqa_items(directory, "train"):
        history = item.problem.history
        earlier = "; ".join(f"{q} (answer: {a})" for q, a in history)
        question = (
            f"{item.problem.question} [after: {earlier}]" if earlier else item.problem.question
        )
        exemplars.append(
            Exemplar(
                id=item.problem.id,
                question=question,
                facts=(),
                solution=item.solution,
                group=item.problem.group,
                query=" / ".join([*(q for q, _ in history), item.problem.question]),
            )
        )
    return exemplars


# --- TAT-QA --------------------------------------------------------------------------


def _tatqa_contexts(directory: Path, split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = json.loads((directory / TATQA_FILES[split]).read_text())
    return records


def _tatqa_problem(context: dict[str, Any], question: dict[str, Any]) -> Problem:
    paragraphs = sorted(context["paragraphs"], key=lambda p: int(p.get("order", 0)))
    return Problem(
        id=str(question["uid"]),
        question=str(question["question"]),
        pre_text=[],
        post_text=[str(p["text"]) for p in paragraphs],
        table=[[str(cell) for cell in row] for row in context["table"]["table"]],
        group=str(context["table"]["uid"]),
    )


def _scale_line(scale: str) -> str:
    return f"SCALE: {scale or 'none'}"


def tatqa_solution(question: dict[str, Any]) -> str | None:
    """Gold reply lines; arithmetic derivations become programs that reproduce the answer."""
    kind, scale, answer = question["answer_type"], question["scale"], question["answer"]
    head = f"TYPE: {kind}\n{_scale_line(scale)}"
    if kind in {"span", "multi-span"}:
        return f"{head}\nANSWER: {' | '.join(str(a) for a in answer)}"
    if kind == "count":
        return f"{head}\nANSWER: {int(float(answer))}"
    program = typed.expression_program(str(question.get("derivation", "")))
    if program is None:
        return None
    for candidate in (program, f"{program}, multiply(#{program.count('(') - 1}, const_100)"):
        result = official.execute(candidate, [])
        if isinstance(result.value, float) and round(result.value, 2) == round(float(answer), 2):
            try:
                return f"{head}\n{program_line(candidate)}"
            except ProgramParseError:
                return None  # Longer than the reader's program bound.
    return None


def tatqa_items(directory: Path, split: str) -> list[Item]:
    items: list[Item] = []
    for context in _tatqa_contexts(directory, split):
        for question in context["questions"]:
            answer = question["answer"]
            listed = answer if isinstance(answer, list) else [answer]
            items.append(
                Item(
                    _tatqa_problem(context, question),
                    question,
                    tatqa_solution(question) or "TYPE: span\nSCALE: none\nANSWER: (no replay)",
                    str(context["table"]["uid"]),
                    f"ANSWER: {' | '.join(str(a) for a in listed)}\n"
                    f"{_scale_line(question['scale'])}",
                )
            )
    return items


def tatqa_exemplars(directory: Path) -> list[Exemplar]:
    exemplars = []
    for context in _tatqa_contexts(directory, "train"):
        for question in context["questions"]:
            solution = tatqa_solution(question)
            if solution is None:
                continue  # Derivation could not be reproduced as a program.
            exemplars.append(
                Exemplar(
                    id=str(question["uid"]),
                    question=str(question["question"]),
                    facts=(),
                    solution=solution,
                    group=str(context["table"]["uid"]),
                )
            )
    return exemplars


# --- Scoring -------------------------------------------------------------------------


def finqa_metrics(value: Any, gold: Any) -> dict[str, float]:
    return {
        "official_accuracy": float(official.official_correct(value, gold)),
        "tolerant_accuracy": float(official.tolerant_correct(value, gold)),
        "lenient_accuracy": float(official.lenient_correct(value, gold)),
    }


def tatqa_metrics(value: Any, gold: Any) -> dict[str, float]:
    from ledger_rag.evaluation.official import tatqa  # SciPy/pandas: the scoring group.

    if value is None:
        return {"exact_match": 0.0, "f1": 0.0}
    decoded = typed.decode(value)
    answer: Any = decoded["answer"]
    if decoded["type"] in {"arithmetic", "count"}:
        answer = answer[0]
    exact, f1 = tatqa.score(gold, answer, decoded["scale"])
    return {"exact_match": exact, "f1": f1}


@dataclasses.dataclass(frozen=True)
class Benchmark:
    name: str
    commit: str
    items: Callable[[Path, str], list[Item]]
    exemplars: Callable[[Path], list[Exemplar]]
    ledger_task: Task
    raw_task: Task
    metrics: Callable[[Any, Any], dict[str, float]]
    primary: str
    files: Callable[[Path, str], Path]


BENCHMARKS: dict[str, Benchmark] = {
    "finqa": Benchmark(
        "finqa",
        "0f16e2867befa6840783e58be38c9efb9229d742",
        finqa_items,
        finqa_exemplars,
        ProgramTask(official_executor),
        direct.DirectNumberTask(),
        finqa_metrics,
        "official_accuracy",
        lambda directory, split: directory / f"{split}.json",
    ),
    "convfinqa": Benchmark(
        "convfinqa",
        "cf3eed2d5984960bf06bb8145bcea5e80b0222a6",
        convfinqa_items,
        convfinqa_exemplars,
        ProgramTask(official_executor),
        direct.DirectNumberTask(),
        finqa_metrics,
        "official_accuracy",
        lambda directory, split: directory / "data.zip",
    ),
    "tatqa": Benchmark(
        "tatqa",
        "870accc41953dcde885aabeb963d94aabdc0fbc3",
        tatqa_items,
        tatqa_exemplars,
        typed.TypedAnswerTask(official_executor),
        direct.DirectTypedTask(),
        tatqa_metrics,
        "exact_match",
        lambda directory, split: directory / TATQA_FILES[split],
    ),
}


# --- Pipelines -----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Pipeline:
    """One arm of a comparison. Every arm of a run uses the same model and mode."""

    name: str
    ledger: bool
    """Ledger contract (programs or typed answers, grounding, execution) or raw answers."""
    samples: int
    exemplars: int
    refine_samples: int = 0

    def describe(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


PIPELINES = ("raw", "ledger", "ledger-zero-shot", "ledger-refine")
REFINE_SAMPLES = 3


def pipeline(name: str, samples: int, exemplars: int) -> Pipeline:
    """``raw``: the model alone, one answer. ``ledger``: the full backend.
    ``ledger-zero-shot``: the backend without retrieved exemplars (an ablation).
    ``ledger-refine``: the backend plus execution-feedback refinement when votes split."""
    if name == "raw":
        return Pipeline("raw", False, 1, 0)
    if name == "ledger":
        return Pipeline("ledger", True, samples, exemplars)
    if name == "ledger-zero-shot":
        return Pipeline("ledger-zero-shot", True, samples, 0)
    if name == "ledger-refine":
        return Pipeline("ledger-refine", True, samples, exemplars, REFINE_SAMPLES)
    raise ValueError(f"unknown pipeline {name!r}; choose from {', '.join(PIPELINES)}")


# --- Readers -------------------------------------------------------------------------


@dataclasses.dataclass
class ReplayGold:
    """Offline reader that replies with each problem's gold solution."""

    solutions: dict[str, str]

    @classmethod
    def of(cls, items: Sequence[Item], *, raw: bool = False) -> "ReplayGold":
        return cls({problem_text(i.problem): i.raw_solution if raw else i.solution for i in items})

    async def read(self, messages: list[dict[str, str]], sample: int) -> Reply:
        key = PAGE_MARKER + messages[-1]["content"].rsplit(PAGE_MARKER, 1)[-1]
        return Reply(text=f"FACTS:\n- replay\n{self.solutions[key]}")


@dataclasses.dataclass
class ChatReader:
    """Adapts the evaluation chat client: sample 0 is the arm, others are sampled."""

    client: ChatClient
    config: GenerationConfig
    sample_temperature: float

    async def read(self, messages: list[dict[str, str]], sample: int) -> Reply:
        config = self.config
        if sample:
            config = dataclasses.replace(
                config,
                temperature=self.sample_temperature,
                seed=(config.seed or 0) + sample,
            )
        completion = await self.client.complete(config, messages)
        return Reply(
            text=completion.text,
            error=completion.error,
            latency_ms=completion.latency_ms,
            completion_tokens=completion.usage.get("completion_tokens", 0),
        )


def model_spec(args: argparse.Namespace) -> models.ModelSpec:
    if args.model_spec:
        return models.ModelSpec.from_manifest(json.loads(Path(args.model_spec).read_text()))
    return models.load(args.model, args.models_dir)


def chat_client(spec: models.ModelSpec, args: argparse.Namespace) -> ChatClient:
    """Self-hosted models are reached at ``--llm-base-url``; hosted ones via their env."""
    if spec.kind == "vllm":
        # A batch run queues thousands of requests on one server; a long free-text
        # answer can wait minutes, and a timeout must never be scored as a wrong answer.
        return ChatClient(
            args.llm_base_url,
            os.environ.get("LEDGER_LLM_SERVE_API_KEY", "local-unused"),
            concurrency=args.llm_concurrency,
            endpoint_kind="self-hosted",
            stream=args.llm_stream,
            request_timeout=args.llm_timeout,
            deadline_seconds=args.llm_deadline,
        )
    base_url, key = os.environ.get(spec.base_url_env), os.environ.get(spec.key_env)
    if not base_url or not key:
        raise SystemExit(f"{spec.id}: set {spec.base_url_env} and {spec.key_env} (ledgerctl run)")
    return ChatClient(
        base_url,
        key,
        concurrency=min(args.llm_concurrency, 8),
        endpoint_kind="hosted",
        stream=args.llm_stream,
        requests_per_minute=spec.requests_per_minute,
    )


# --- Reporting -----------------------------------------------------------------------


def select(items: list[Item], limit: int | None, seed: int) -> list[Item]:
    """All items, or a seeded uniform sample of ``limit`` (kept in file order)."""
    if limit is None or limit >= len(items):
        return items
    chosen = set(random.Random(seed).sample(range(len(items)), limit))
    return [item for index, item in enumerate(items) if index in chosen]


def consistency_curve(
    rows: list[dict[str, Any]], items: Sequence[Item], benchmark: Benchmark
) -> dict[str, float]:
    """Primary metric when voting over only the first k samples (k=1 is greedy)."""
    total = max((len(row["sample_values"]) for row in rows), default=0)
    curve: dict[str, float] = {}
    for k in range(1, total + 1):
        hits = 0.0
        for row, item in zip(rows, items, strict=True):
            votes = [Candidate(None, value) for value in row["sample_values"][:k]]
            hits += benchmark.metrics(decide(votes).value, item.gold)[benchmark.primary]
        curve[str(k)] = round(hits / len(rows), 4) if rows else 0.0
    return curve


def summarize(
    rows: list[dict[str, Any]], items: Sequence[Item], benchmark: Benchmark, *, seed: int
) -> dict[str, Any]:
    clusters = [item.cluster for item in items]
    reasons: Counter[str] = Counter(reason for row in rows for reason in row["reasons"])
    scored: list[dict[str, float | None]] = [
        {**row["metrics"], "answered": float(row["value"] is not None)} for row in rows
    ]
    return {
        "metrics": summarize_rows(scored, clusters, seed=seed),
        "sample_outcomes": dict(sorted(reasons.items())),
        "selective": risk_coverage(
            [row["agreement"] for row in rows],
            [row["metrics"][benchmark.primary] == 1.0 for row in rows],
        ),
        "latency": latency([row["latency_ms"] for row in rows]),
        "completion_tokens": sum(row["completion_tokens"] for row in rows),
        "consistency_curve": consistency_curve(rows, items, benchmark),
    }


def paired(
    arms: dict[str, list[dict[str, Any]]], items: Sequence[Item], *, seed: int
) -> dict[str, Any]:
    """Every non-raw arm minus ``raw`` on identical questions, per metric, with CIs."""
    if "raw" not in arms:
        return {}
    clusters = {item.problem.id: item.cluster for item in items}
    out: dict[str, Any] = {}
    for name, rows in arms.items():
        if name == "raw":
            continue
        out[f"{name}-minus-raw"] = (
            {
                metric: paired_cluster_difference(
                    {
                        i.problem.id: r["metrics"][metric]
                        for i, r in zip(items, arms["raw"], strict=True)
                    },
                    {i.problem.id: r["metrics"][metric] for i, r in zip(items, rows, strict=True)},
                    clusters,
                    seed=seed,
                )
                for metric in rows[0]["metrics"]
            }
            if rows
            else {}
        )
    return out


def first_round(row: dict[str, Any], item: Item, benchmark: Benchmark) -> dict[str, Any]:
    """An arm's row as it was before refinement: the vote over its first-round samples."""
    decision = decide([Candidate(None, value) for value in row["sample_values"]])
    return {
        **row,
        "value": decision.value,
        "votes": decision.votes,
        "valid": decision.valid,
        "agreement": round(decision.agreement, 4),
        "refined_reasons": [],
        "metrics": benchmark.metrics(decision.value, item.gold),
        "derived_from": "first round of the refinement arm",
    }


def retrieval_lineage(path: Path) -> dict[str, Any]:
    contexts = retrieved_contexts(path)
    return {
        "contexts_sha256": sha256_file(path),
        "arm": contexts.get("answer_retrieval"),
        "scope": contexts.get("scope"),
        "top_k": contexts.get("top_k"),
        "collection": contexts["lineage"].get("collection"),
        "corpus_sha256": contexts["lineage"].get("corpus_sha256"),
    }


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


async def _solve_arm(
    arm: Pipeline,
    benchmark: Benchmark,
    items: Sequence[Item],
    reader: ReplayGold | ChatReader,
    index: ExemplarIndex | None,
    keep_replies: bool,
) -> list[Solution]:
    def search(problem: Problem) -> list[Exemplar]:
        assert index is not None
        query = " / ".join([*(q for q, _ in problem.history), problem.question])
        return index.search(
            query, arm.exemplars, exclude_group=problem.group, exclude_id=problem.id
        )

    solver = Solver(
        reader=reader,
        task=benchmark.ledger_task if arm.ledger else benchmark.raw_task,
        samples=arm.samples,
        exemplar_search=search if arm.exemplars and index is not None else None,
        keep_replies=keep_replies,
        refine_samples=arm.refine_samples,
    )
    return list(await asyncio.gather(*(solver.solve(item.problem) for item in items)))


async def run(args: argparse.Namespace) -> dict[str, Any]:
    benchmark = BENCHMARKS[args.benchmark]
    directory: Path = args.data_dir or Path("data") / benchmark.name
    verify(directory, benchmark.name)  # Pinned SHA-256 of every artifact, before anything.
    if args.contexts:
        available = finqa_retrieved_items(directory, args.split, args.contexts)
    else:
        available = benchmark.items(directory, args.split)
    items = select(available, args.limit, args.seed)
    arms = [pipeline(name, args.samples, args.exemplars) for name in args.pipelines]
    needs_index = any(arm.exemplars for arm in arms)
    index = ExemplarIndex(benchmark.exemplars(directory)) if needs_index else None

    spec: models.ModelSpec | None = None
    config: GenerationConfig | None = None
    client: ChatClient | None = None
    readers: dict[str, ReplayGold | ChatReader] = {}
    if args.reader == "replay-gold":
        for arm in arms:
            readers[arm.name] = ReplayGold.of(items, raw=not arm.ledger)
    else:
        spec = model_spec(args)
        config = spec.generation(args.mode, seed=args.seed)
        client = chat_client(spec, args)
        for arm in arms:
            readers[arm.name] = ChatReader(client, config, args.sample_temperature)
    started = time.perf_counter()
    try:
        solved = await asyncio.gather(
            *(
                _solve_arm(
                    arm, benchmark, items, readers[arm.name], index, bool(args.debug_replies)
                )
                for arm in arms
            )
        )
    finally:
        if client is not None:
            await client.close()
    rows: dict[str, list[dict[str, Any]]] = {}
    debug: list[dict[str, Any]] = []
    for arm, solutions in zip(arms, solved, strict=True):
        rows[arm.name] = []
        for item, solution in zip(items, solutions, strict=True):
            rows[arm.name].append(
                {
                    "metrics": benchmark.metrics(solution.decision.value, item.gold),
                    **solution.record(),
                }
            )
            invalid = [s for s in solution.samples if s.reason != "ok"]
            if args.debug_replies and invalid and len(debug) < args.debug_replies:
                debug.append(
                    {
                        "arm": arm.name,
                        "id": item.problem.id,
                        "gold": item.solution if arm.ledger else item.raw_solution,
                        "replies": [{"reason": s.reason, "text": s.text} for s in invalid],
                    }
                )
    for arm in arms:
        # Refinement only changes split votes, so the same first round also measures the
        # plain ledger arm without paying for it twice.
        base = arm.name.removesuffix("-refine")
        if arm.refine_samples and base not in rows:
            rows[base] = [
                first_round(row, item, benchmark)
                for row, item in zip(rows[arm.name], items, strict=True)
            ]
    tasks = {arm.name: benchmark.ledger_task if arm.ledger else benchmark.raw_task for arm in arms}
    return {
        "benchmark": benchmark.name,
        "setting": "retrieved" if args.contexts else "gold-evidence",
        "retrieval": retrieval_lineage(args.contexts) if args.contexts else None,
        "split": args.split,
        "questions": len(items),
        "available": len(available),
        "primary": benchmark.primary,
        "reader": args.reader,
        "model": spec.manifest() if spec else None,
        "mode": args.mode or (spec.default_mode if spec else None),
        "generation": config.manifest() if config else None,
        "sample_temperature": args.sample_temperature,
        "pipelines": [arm.describe() for arm in arms],
        "provider_stats": dataclasses.asdict(client.stats) if client else None,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "lineage": {
            "dataset_commit": benchmark.commit,
            "split_sha256": sha256_file(benchmark.files(directory, args.split)),
            "exemplar_source_sha256": sha256_file(benchmark.files(directory, "train")),
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": {
                name: hashlib.sha256(task.system_prompt.encode()).hexdigest()
                for name, task in tasks.items()
            },
            "selection_seed": args.seed,
            "git": git_commit(),
        },
        "arms": {
            name: summarize(arm_rows, items, benchmark, seed=args.seed)
            for name, arm_rows in rows.items()
        },
        "paired": paired(rows, items, seed=args.seed),
        "results": [
            {
                "id": item.problem.id,
                "cluster": item.cluster,
                "turn": len(item.problem.history),
                "arms": {name: arm_rows[index] for name, arm_rows in rows.items()},
            }
            for index, item in enumerate(items)
        ],
        **({"debug_replies": debug} if args.debug_replies else {}),
    }


def _names(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    for name in names:
        if name not in PIPELINES:
            raise argparse.ArgumentTypeError(f"unknown pipeline {name!r}")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), default="finqa")
    parser.add_argument("--data-dir", type=Path, default=None, help="Default: data/<benchmark>")
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--unseal-test", action="store_true", help="Required for --split test")
    parser.add_argument("--limit", type=int, default=None, help="Seeded sample size")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--reader", choices=READERS, default="replay-gold")
    parser.add_argument(
        "--pipelines", type=_names, default=["raw", "ledger"], help=f"Arms: {', '.join(PIPELINES)}"
    )
    parser.add_argument("--samples", type=int, default=5, help="Ledger self-consistency samples")
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--exemplars", type=int, default=4, help="Ledger dynamic exemplars")
    parser.add_argument(
        "--contexts",
        type=Path,
        default=None,
        help="FinQA only: saved retrieval contexts; answer from retrieved chunks, not the page",
    )
    parser.add_argument("--model", default="", help="Registry id in config/models")
    parser.add_argument("--models-dir", type=Path, default=models.MODELS_DIR)
    parser.add_argument("--model-spec", default="", help="A ModelSpec manifest JSON file")
    parser.add_argument("--mode", default=None, help="Model mode; default: the model's default")
    parser.add_argument(
        "--llm-base-url",
        default=os.environ.get("LEDGER_LLM_SERVE_URL", "http://127.0.0.1:8000/v1"),
        help="Endpoint of a self-hosted (vllm) model",
    )
    parser.add_argument("--llm-concurrency", type=int, default=128)
    parser.add_argument(
        "--llm-timeout", type=float, default=900, help="Self-hosted read timeout, seconds"
    )
    parser.add_argument(
        "--llm-deadline", type=float, default=2400, help="Self-hosted per-request deadline"
    )
    parser.add_argument("--llm-stream", action="store_true")
    parser.add_argument(
        "--debug-replies",
        type=int,
        default=0,
        help="Keep raw replies of up to N questions with invalid samples (public data only)",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.split == "test" and not args.unseal_test:
        parser.error("the test split is sealed; pass --unseal-test only with a dev-frozen config")
    if args.contexts and args.benchmark != "finqa":
        parser.error("--contexts applies to FinQA only")
    if args.benchmark == "convfinqa" and args.split == "test":
        parser.error("ConvFinQA test answers are private; report dev")
    if args.split == "train" and args.exemplars and "ledger" in args.pipelines:
        parser.error("train questions would retrieve themselves; use --exemplars 0")
    if args.reader == "endpoint" and not (args.model or args.model_spec):
        parser.error("--reader endpoint needs --model (or --model-spec)")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not 1 <= args.samples <= 32:
        parser.error("--samples must be within [1, 32]")


def brief(report: dict[str, Any]) -> str:
    primary = report["primary"]
    model = (report.get("model") or {}).get("id", report["reader"])
    parts = [f"{report['benchmark']}/{report['split']} n={report['questions']} model={model}"]
    for name, arm in report["arms"].items():
        metric = arm["metrics"][primary]
        parts.append(
            f"{name}:{primary}={metric['mean']} [{metric['ci95'][0]}, {metric['ci95'][1]}]"
        )
    for name, deltas in report["paired"].items():
        delta = deltas.get(primary)
        if delta:
            parts.append(f"{name}={delta['delta']} [{delta['ci95'][0]}, {delta['ci95'][1]}]")
    return " ".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate(parser, args)
    report = asyncio.run(run(args))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(brief(report), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
