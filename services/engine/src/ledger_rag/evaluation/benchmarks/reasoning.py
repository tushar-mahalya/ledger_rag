"""FinQA program-of-thought benchmark in the standard (gold page) setting.

Every question is answered from its own report page (pre-text, table, post-text), as
in the FinQA paper and every leaderboard entry, so scores are comparable with
published execution accuracy. Open-corpus retrieval is measured separately
(:mod:`ledger_rag.evaluation.benchmarks.finqa`); do not mix the two settings.

Pipeline per question: dynamic exemplars from the training split -> N sampled DSL
programs -> parse, ground, execute with the official executor -> majority vote.
Scores: official execution accuracy (exact, as the official ``evaluate.py``) and the
Program-of-Thoughts tolerant comparison, each with report-cluster bootstrap CIs.

Readers:

- ``replay-gold``: answers with the gold program; an offline pipeline check that
  must score 1.0. It needs no endpoint and spends nothing.
- ``endpoint``: an OpenAI-compatible chat endpoint (vLLM on Modal, Gemini, ...).

The held-out test split is sealed: ``--split test`` requires ``--unseal-test`` and
must only be run with a configuration frozen on dev.

Usage::

    python -m ledger_rag.evaluation.benchmarks.finqa_reasoning --reader replay-gold
    python -m ledger_rag.evaluation.benchmarks.finqa_reasoning --reader endpoint \\
        --llm-model Qwen/Qwen3.8-27B-FP8 --samples 5 --exemplars 4 --output run.json
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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ledger_rag.evaluation.datasets.fetch import verify
from ledger_rag.evaluation.llm import ENDPOINT_KINDS, ChatClient, GenerationConfig
from ledger_rag.evaluation.official import finqa as official
from ledger_rag.evaluation.uncertainty import latency, risk_coverage, summarize_rows
from ledger_rag.reasoning.consistency import Candidate, decide
from ledger_rag.reasoning.dsl import Program
from ledger_rag.reasoning.exemplars import Exemplar, ExemplarIndex
from ledger_rag.reasoning.solver import (
    PAGE_MARKER,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    Execution,
    Problem,
    Reply,
    Solver,
    problem_text,
)

BENCHMARKS = ("finqa", "convfinqa")
DATASET_COMMITS = {
    "finqa": "0f16e2867befa6840783e58be38c9efb9229d742",
    "convfinqa": "cf3eed2d5984960bf06bb8145bcea5e80b0222a6",
}
SPLITS = ("train", "dev", "test")
READERS = ("replay-gold", "endpoint")
CONVFINQA_FILES = {"train": "data/train.json", "dev": "data/dev.json"}


@dataclasses.dataclass(frozen=True)
class Item:
    """One scored question: the problem, its gold answer/program and bootstrap cluster."""

    problem: Problem
    gold: float | str
    program: str
    cluster: str


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


def _page(record: dict[str, Any], **fields: Any) -> Problem:
    return Problem(
        pre_text=list(record["pre_text"]),
        post_text=list(record["post_text"]),
        table=[list(row) for row in record["table"]],
        group=str(record["filename"]),
        **fields,
    )


def finqa_items(directory: Path, split: str) -> list[Item]:
    return [
        Item(
            _page(record, id=str(record["id"]), question=str(record["qa"]["question"])),
            record["qa"]["exe_ans"],
            str(record["qa"]["program"]),
            report_cluster(str(record["filename"])),
        )
        for record in load_split(directory, split)
    ]


def finqa_exemplars(directory: Path) -> list[Exemplar]:
    return [
        Exemplar(
            id=str(record["id"]),
            question=str(record["qa"]["question"]),
            facts=tuple(str(t).strip() for t in record["qa"].get("gold_inds", {}).values()),
            program=str(record["qa"]["program"]),
            group=str(record["filename"]),
        )
        for record in load_split(directory, "train")
    ]


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
            items.append(
                Item(
                    _page(
                        record,
                        id=f"{record['id']}/turn_{turn}",
                        question=str(question),
                        history=history,
                    ),
                    answers[turn],
                    str(programs[turn]),
                    report_cluster(str(record["filename"])),
                )
            )
    return items


def convfinqa_exemplars(directory: Path) -> list[Exemplar]:
    """Train turns; the retrieval text carries earlier questions so follow-ups match."""
    return [
        Exemplar(
            id=item.problem.id,
            question=" / ".join([*(q for q, _ in item.problem.history), item.problem.question]),
            facts=(),
            program=item.program,
            group=item.problem.group,
        )
        for item in convfinqa_items(directory, "train")
    ]


LOADERS = {
    "finqa": (finqa_items, finqa_exemplars),
    "convfinqa": (convfinqa_items, convfinqa_exemplars),
}


def official_executor(program: Program, item: Problem) -> Execution:
    result = official.execute(program.text, item.table)
    return Execution(result.value, result.valid)


@dataclasses.dataclass
class ReplayGold:
    """Offline reader that replies with each problem's gold program."""

    programs: dict[str, str]

    @classmethod
    def of(cls, items: Sequence[Item]) -> "ReplayGold":
        return cls({problem_text(item.problem): item.program for item in items})

    async def read(self, messages: list[dict[str, str]], sample: int) -> Reply:
        key = PAGE_MARKER + messages[-1]["content"].rsplit(PAGE_MARKER, 1)[-1]
        return Reply(text=f"FACTS:\n- replay\nPROGRAM: {self.programs[key]}")


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


def select(items: list[Item], limit: int | None, seed: int) -> list[Item]:
    """All items, or a seeded uniform sample of ``limit`` (kept in file order)."""
    if limit is None or limit >= len(items):
        return items
    chosen = set(random.Random(seed).sample(range(len(items)), limit))
    return [item for index, item in enumerate(items) if index in chosen]


def score_rows(results: list[dict[str, Any]]) -> list[dict[str, float | None]]:
    return [
        {
            "official_accuracy": float(row["official"]),
            "tolerant_accuracy": float(row["tolerant"]),
            "answered": float(row["value"] is not None),
        }
        for row in results
    ]


def consistency_curve(results: list[dict[str, Any]], golds: list[Any]) -> dict[str, float]:
    """Official accuracy when voting over only the first k samples (k=1 is greedy)."""
    total = max((len(row["sample_values"]) for row in results), default=0)
    curve: dict[str, float] = {}
    for k in range(1, total + 1):
        hits = 0
        for row, gold in zip(results, golds, strict=True):
            votes = [Candidate(None, value) for value in row["sample_values"][:k]]
            hits += official.official_correct(decide(votes).value, gold)
        curve[str(k)] = round(hits / len(results), 4) if results else 0.0
    return curve


def summarize(results: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    clusters = [row["cluster"] for row in results]
    reasons: Counter[str] = Counter(reason for row in results for reason in row["reasons"])
    return {
        "metrics": summarize_rows(score_rows(results), clusters, seed=seed),
        "sample_outcomes": dict(sorted(reasons.items())),
        "selective": risk_coverage(
            [row["agreement"] for row in results], [row["official"] for row in results]
        ),
        "latency": latency([row["latency_ms"] for row in results]),
        "completion_tokens": sum(row["completion_tokens"] for row in results),
    }


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _split_file(benchmark: str, directory: Path, split: str) -> Path:
    return directory / ("data.zip" if benchmark == "convfinqa" else f"{split}.json")


async def run(args: argparse.Namespace) -> dict[str, Any]:
    benchmark: str = args.benchmark
    directory: Path = args.data_dir or Path("data") / benchmark
    verify(directory, benchmark)  # Pinned SHA-256 of every artifact, before anything runs.
    load_items, load_exemplars = LOADERS[benchmark]
    available = load_items(directory, args.split)
    items = select(available, args.limit, args.seed)
    index = ExemplarIndex(load_exemplars(directory)) if args.exemplars else None

    def search(item: Problem) -> list[Exemplar]:
        assert index is not None
        query = " / ".join([*(q for q, _ in item.history), item.question])
        return index.search(query, args.exemplars, exclude_group=item.group, exclude_id=item.id)

    client: ChatClient | None = None
    config: GenerationConfig | None = None
    if args.reader == "replay-gold":
        reader: ReplayGold | ChatReader = ReplayGold.of(items)
    else:
        config = GenerationConfig(
            model=args.llm_model,
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
            reasoning_effort=args.llm_reasoning_effort,
            seed=args.llm_seed,
            extra_body=json.loads(args.llm_extra_body) if args.llm_extra_body else {},
        )
        client = ChatClient(
            args.llm_base_url,
            args.llm_key,
            concurrency=args.llm_concurrency,
            endpoint_kind=args.llm_endpoint_kind,
            stream=args.llm_stream,
            requests_per_minute=args.llm_rpm,
        )
        reader = ChatReader(client, config, args.sample_temperature)
    solver = Solver(
        reader=reader,
        executor=official_executor,
        samples=args.samples,
        exemplar_search=search if index is not None else None,
        keep_replies=bool(args.debug_replies),
    )
    started = time.perf_counter()
    try:
        solutions = await asyncio.gather(*(solver.solve(item.problem) for item in items))
    finally:
        if client is not None:
            await client.close()
    results = []
    for item, solution in zip(items, solutions, strict=True):
        value = solution.decision.value
        results.append(
            {
                "id": item.problem.id,
                "cluster": item.cluster,
                "turn": len(item.problem.history),
                "official": official.official_correct(value, item.gold),
                "tolerant": official.tolerant_correct(value, item.gold),
                **solution.record(),
            }
        )
    debug: list[dict[str, Any]] = []
    if args.debug_replies:
        for item, solution in zip(items, solutions, strict=True):
            invalid = [s for s in solution.samples if s.reason != "ok"]
            if invalid and len(debug) < args.debug_replies:
                debug.append(
                    {
                        "id": item.problem.id,
                        "gold_program": item.program,
                        "replies": [{"reason": s.reason, "text": s.text} for s in invalid],
                    }
                )
    summary = summarize(results, seed=args.seed)
    summary["consistency_curve"] = consistency_curve(results, [item.gold for item in items])
    return {
        "benchmark": benchmark,
        "setting": "gold-page",
        "split": args.split,
        "questions": len(results),
        "available": len(available),
        "reader": args.reader,
        "samples": args.samples,
        "sample_temperature": args.sample_temperature,
        "exemplars": args.exemplars,
        "generation": config.manifest() if config else None,
        "provider_stats": dataclasses.asdict(client.stats) if client else None,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "lineage": {
            "dataset_commit": DATASET_COMMITS[benchmark],
            "split_sha256": sha256_file(_split_file(benchmark, directory, args.split)),
            "exemplar_source_sha256": sha256_file(_split_file(benchmark, directory, "train")),
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "selection_seed": args.seed,
            "git": git_commit(),
        },
        "summary": summary,
        "results": results,
        **({"debug_replies": debug} if args.debug_replies else {}),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="finqa")
    parser.add_argument("--data-dir", type=Path, default=None, help="Default: data/<benchmark>")
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--unseal-test", action="store_true", help="Required for --split test")
    parser.add_argument("--limit", type=int, default=None, help="Seeded sample size")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--reader", choices=READERS, default="replay-gold")
    parser.add_argument("--samples", type=int, default=1, help="Self-consistency samples")
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--exemplars", type=int, default=4, help="Dynamic exemplars (0: none)")
    parser.add_argument(
        "--llm-base-url",
        default=os.environ.get("LEDGER_LLM_BASE_URL", "http://localhost:8000/v1"),
    )
    parser.add_argument("--llm-key", default=os.environ.get("LEDGER_LLM_API_KEY", "local-unused"))
    parser.add_argument("--llm-model", default=os.environ.get("LEDGER_LLM_MODEL", ""))
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-max-tokens", type=int, default=1024)
    parser.add_argument("--llm-reasoning-effort", default=None)
    parser.add_argument("--llm-seed", type=int, default=0)
    parser.add_argument("--llm-extra-body", default=None, help="JSON merged into each request")
    parser.add_argument("--llm-endpoint-kind", choices=ENDPOINT_KINDS, default="hosted")
    parser.add_argument("--llm-concurrency", type=int, default=8)
    parser.add_argument("--llm-rpm", type=float, default=None, help="Hosted pacing")
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
    if args.benchmark == "convfinqa" and args.split == "test":
        parser.error("ConvFinQA test answers are private; report dev")
    if args.split == "train" and args.exemplars:
        parser.error("train questions would retrieve themselves; use --exemplars 0")
    if args.reader == "endpoint" and not args.llm_model:
        parser.error("--reader endpoint needs --llm-model")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")


def brief(report: dict[str, Any]) -> str:
    metrics = report["summary"]["metrics"]
    parts = [f"{report['benchmark']}/{report['split']} n={report['questions']}"]
    for name in ("official_accuracy", "tolerant_accuracy"):
        metric = metrics[name]
        parts.append(f"{name}={metric['mean']} [{metric['ci95'][0]}, {metric['ci95'][1]}]")
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
