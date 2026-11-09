"""Program-of-thought solver: sample replies, judge each, vote on the outcome.

The solver is model-agnostic. A :class:`Reader` turns messages into one reply per
sample index (index 0 is the deterministic arm, later indices are stochastic
samples); benchmarks adapt the evaluation chat client, tests use doubles.
A :class:`Task` is the answer contract: its system prompt and how one reply is
parsed, grounded and executed into a voting candidate. :class:`ProgramTask`
(FinQA/ConvFinQA) takes an injected executor so benchmarks use the official FinQA
semantics while production can use the unit-aware Decimal executor.
"""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ledger_rag.reasoning.consistency import Candidate, Decision, decide
from ledger_rag.reasoning.dsl import Program, ProgramParseError, parse
from ledger_rag.reasoning.exemplars import Exemplar, render
from ledger_rag.reasoning.grounding import Evidence, repair_rows, ungrounded

PROMPT_VERSION = "pot-dsl-v3"
SYSTEM_PROMPT = """You answer numerical questions about a page of a company's annual report.
Do not calculate anything yourself: write a program, and a calculator will execute it.

Operations, each with exactly two operands:
  add(a, b)  subtract(a, b)  multiply(a, b)  divide(a, b)  exp(a, b)
  greater(a, b)  -> yes if a > b, else no
  table_sum(ROW, none)  table_average(ROW, none)  table_max(ROW, none)  table_min(ROW, none)
    ROW is the exact text of a table row's first cell; these aggregate that row's numbers.
Operands: a number copied exactly from the page or question (no $ or thousands separators;
keep a trailing % if the page shows one), a constant (const_2, const_3, ... for counts such
as the number of years averaged, const_100, const_1000, const_1000000, const_m1 for -1)
or #k, the result of step k (#0 is the first step; never nest operations).
The result of the last step is the answer. In a conversation, answer only the last question;
earlier answers are context. Follow the conventions of the solved examples, for example
whether a percentage is left as a ratio or multiplied by const_100.

Reply in exactly this format and nothing after it:
FACTS:
- <each figure you use and what it measures>
PROGRAM: <steps separated by ", ">"""


@dataclass(frozen=True)
class Reply:
    text: str
    error: str | None = None
    latency_ms: float = 0.0
    completion_tokens: int = 0


class Reader(Protocol):
    async def read(self, messages: list[dict[str, str]], sample: int) -> Reply:
        """Sample 0 is the configured (usually greedy) arm; later samples are stochastic."""
        ...


class Task(Protocol):
    """An answer contract: the instructions and the judge of one reply."""

    @property
    def system_prompt(self) -> str: ...

    def judge(self, text: str, problem: "Problem", evidence: Evidence) -> Candidate: ...


@dataclass(frozen=True)
class Problem:
    id: str
    question: str
    pre_text: Sequence[str]
    post_text: Sequence[str]
    table: Sequence[Sequence[str]]
    group: str = ""
    """Exemplars from the same group (report page) are excluded."""
    history: Sequence[tuple[str, str]] = ()
    """Earlier (question, answer) turns of a conversation, oldest first."""


@dataclass(frozen=True)
class Execution:
    value: float | str | None
    valid: bool


Executor = Callable[[Program, Problem], Execution]


def _cell(text: str) -> str:
    return " ".join(text.replace("|", "/").split())


def render_table(table: Sequence[Sequence[str]]) -> str:
    if not table:
        return ""
    width = max(len(row) for row in table)
    rows = [[_cell(c) for c in row] + [""] * (width - len(row)) for row in table]
    lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([lines[0], "|" + "---|" * width, *lines[1:]])


def _sentences(texts: Sequence[str]) -> str:
    return " ".join(t.strip() for t in texts if t.strip() and t.strip() != ".")


def render_page(problem: Problem) -> str:
    parts = [
        _sentences(problem.pre_text),
        render_table(problem.table),
        _sentences(problem.post_text),
    ]
    return "\n\n".join(part for part in parts if part)


PAGE_MARKER = "Page:\n"


def problem_text(problem: Problem) -> str:
    """The page, any conversation so far and the question: everything but exemplars."""
    history = "".join(f"Q: {question}\nA: {answer}\n" for question, answer in problem.history)
    conversation = f"\n\nConversation so far:\n{history}" if history else ""
    return f"{PAGE_MARKER}{render_page(problem)}{conversation}\n\nQuestion: {problem.question}"


def messages(
    problem: Problem, exemplars: Sequence[Exemplar], system_prompt: str = SYSTEM_PROMPT
) -> list[dict[str, str]]:
    shots = render(exemplars)
    prefix = f"Solved examples:\n\n{shots}\n\n---\n\n" if shots else ""
    content = prefix + problem_text(problem)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


@dataclass(frozen=True)
class Sample:
    program: str | None
    value: float | str | None
    reason: str
    latency_ms: float
    completion_tokens: int
    error: str | None = None
    text: str | None = None
    """Raw reply, kept only when the solver runs with ``keep_replies`` (diagnostics)."""


@dataclass(frozen=True)
class Solution:
    decision: Decision
    samples: tuple[Sample, ...]
    exemplar_ids: tuple[str, ...] = field(default=())

    def record(self) -> dict[str, Any]:
        """Report fields only: no prompt, page text or raw model reply."""
        return {
            "program": self.decision.program,
            "value": self.decision.value,
            "votes": self.decision.votes,
            "valid": self.decision.valid,
            "samples": self.decision.total,
            "agreement": round(self.decision.agreement, 4),
            "reasons": [sample.reason for sample in self.samples],
            "sample_values": [sample.value for sample in self.samples],
            "exemplar_ids": list(self.exemplar_ids),
            "latency_ms": max((s.latency_ms for s in self.samples), default=0.0),
            "completion_tokens": sum(s.completion_tokens for s in self.samples),
        }


def run_program(text: str, problem: Problem, evidence: Evidence, executor: Executor) -> Candidate:
    """Parse, ground and execute a program reply into a voting candidate."""
    try:
        program = repair_rows(parse(text), evidence)
    except ProgramParseError:
        return Candidate(None, None, "parse")
    if ungrounded(program, evidence):
        return Candidate(program.text, None, "ungrounded")
    execution = executor(program, problem)
    if not execution.valid or execution.value is None:
        return Candidate(program.text, None, "execution")
    return Candidate(program.text, execution.value, "ok")


@dataclass(frozen=True)
class ProgramTask:
    """FinQA/ConvFinQA contract: the answer is the executed program."""

    executor: Executor
    system_prompt: str = SYSTEM_PROMPT

    def judge(self, text: str, problem: Problem, evidence: Evidence) -> Candidate:
        return run_program(text, problem, evidence, self.executor)


@dataclass
class Solver:
    reader: Reader
    task: Task
    samples: int = 1
    exemplar_search: Callable[[Problem], list[Exemplar]] | None = None
    keep_replies: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.samples <= 32:
            raise ValueError("samples must be within [1, 32]")

    async def solve(self, problem: Problem) -> Solution:
        shots = self.exemplar_search(problem) if self.exemplar_search else []
        prompt = messages(problem, shots, self.task.system_prompt)
        evidence = Evidence.build(
            [*problem.pre_text, *problem.post_text], problem.table, problem.question
        )
        replies = await asyncio.gather(
            *(self.reader.read(prompt, index) for index in range(self.samples))
        )
        samples: list[Sample] = []
        candidates: list[Candidate] = []
        for reply in replies:
            if reply.error:
                candidate = Candidate(None, None, "provider")
            else:
                candidate = self.task.judge(reply.text, problem, evidence)
            candidates.append(candidate)
            samples.append(
                Sample(
                    program=candidate.program,
                    value=candidate.value,
                    reason=candidate.reason,
                    latency_ms=reply.latency_ms,
                    completion_tokens=reply.completion_tokens,
                    error=reply.error,
                    text=reply.text if self.keep_replies else None,
                )
            )
        return Solution(decide(candidates), tuple(samples), tuple(e.id for e in shots))
