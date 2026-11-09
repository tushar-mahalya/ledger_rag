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
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from ledger_rag.reasoning.consistency import Candidate, Decision, decide
from ledger_rag.reasoning.dsl import Program, ProgramParseError
from ledger_rag.reasoning.exemplars import Exemplar, render
from ledger_rag.reasoning.expression import read_program
from ledger_rag.reasoning.grounding import Evidence, repair_rows, ungrounded

PROMPT_VERSION = "pot-expression-v4"
SYSTEM_PROMPT = """You answer numerical questions about a page of a company's annual report.
Do not calculate anything yourself: write one arithmetic expression, and a calculator
will evaluate it.

The expression uses numbers copied exactly from the page or question (keep a % sign if
the page shows one), + - * / and ** with parentheses, and optionally:
  table_sum("ROW")  table_average("ROW")  table_max("ROW")  table_min("ROW")
    which aggregate the numbers of the table row whose first cell is exactly ROW;
  a > b  for yes/no comparisons.
Counts such as the number of years averaged and unit factors (100, 1000) may be
written directly. If the answer is a single figure on the page, write just that number.
In a conversation, answer only the last question; earlier answers can be reused.
Follow the conventions of the solved examples, for example whether a percentage is
left as a ratio or multiplied by 100.

Reply in exactly this format and nothing after it:
FACTS:
- <each figure you use and what it measures>
PROGRAM: <one expression, e.g. (5829 - 5735) / 5735>"""


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
    """First-round samples (what the self-consistency curve is computed from)."""
    exemplar_ids: tuple[str, ...] = field(default=())
    refined: tuple[Sample, ...] = field(default=())
    """Execution-feedback replies, when refinement ran."""

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
            "refined_reasons": [sample.reason for sample in self.refined],
            "latency_ms": max((s.latency_ms for s in self.samples), default=0.0)
            + max((s.latency_ms for s in self.refined), default=0.0),
            "completion_tokens": sum(s.completion_tokens for s in (*self.samples, *self.refined)),
        }


def run_program(text: str, problem: Problem, evidence: Evidence, executor: Executor) -> Candidate:
    """Parse, ground and execute a program reply into a voting candidate."""
    try:
        program = repair_rows(read_program(text), evidence)
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


_FAILURES = {
    "parse": "could not be read in the required format",
    "ungrounded": "uses numbers, rows or text that are not in the evidence",
    "execution": "could not be executed (for example division by zero or an unknown row)",
    "provider": "was not produced",
    "type": "gave yes/no where a number was needed",
}


def _shown(candidate: Candidate) -> str:
    if candidate.program:
        return candidate.program
    if isinstance(candidate.value, str) and candidate.value.startswith("{"):
        return candidate.value  # A typed answer: its canonical JSON.
    return "(unreadable)"


def feedback(candidates: Sequence[Candidate]) -> str:
    """Execution feedback for a refinement round: each attempt and what it produced."""
    lines = []
    for number, candidate in enumerate(candidates, 1):
        result = (
            f"= {candidate.value}"
            if candidate.value is not None
            else _FAILURES.get(candidate.reason, "was invalid")
        )
        lines.append(f"{number}. {_shown(candidate)} -> {result}")
    return (
        "Independent attempts at this question and what the calculator returned:\n"
        + "\n".join(lines)
        + "\n\nThey disagree or failed. Re-read the question and the evidence, decide which "
        "figures, units and operations are right, and reply in the same format with one "
        "corrected answer."
    )


@dataclass
class Solver:
    reader: Reader
    task: Task
    samples: int = 1
    exemplar_search: Callable[[Problem], list[Exemplar]] | None = None
    keep_replies: bool = False
    refine_samples: int = 0
    """Execution-feedback refinement: when no answer wins a strict majority, this many
    informed replies (each counted twice) join the vote. 0 disables it."""

    def __post_init__(self) -> None:
        if not 1 <= self.samples <= 32:
            raise ValueError("samples must be within [1, 32]")
        if not 0 <= self.refine_samples <= 8:
            raise ValueError("refine_samples must be within [0, 8]")

    async def _round(
        self, prompt: list[dict[str, str]], count: int, problem: Problem, evidence: Evidence
    ) -> list[tuple[Candidate, Sample]]:
        replies = await asyncio.gather(*(self.reader.read(prompt, i) for i in range(count)))
        out = []
        for reply in replies:
            if reply.error:
                candidate = Candidate(None, None, "provider")
            else:
                candidate = self.task.judge(reply.text, problem, evidence)
            sample = Sample(
                program=candidate.program,
                value=candidate.value,
                reason=candidate.reason,
                latency_ms=reply.latency_ms,
                completion_tokens=reply.completion_tokens,
                error=reply.error,
                text=reply.text if self.keep_replies else None,
            )
            out.append((candidate, sample))
        return out

    async def solve(self, problem: Problem) -> Solution:
        shots = self.exemplar_search(problem) if self.exemplar_search else []
        prompt = messages(problem, shots, self.task.system_prompt)
        earlier = [f"{question} {answer}" for question, answer in problem.history]
        evidence = Evidence.build(
            [*problem.pre_text, *problem.post_text, *earlier], problem.table, problem.question
        )
        first = await self._round(prompt, self.samples, problem, evidence)
        candidates = [candidate for candidate, _ in first]
        decision = decide(candidates)
        refined: list[tuple[Candidate, Sample]] = []
        if self.refine_samples and decision.votes * 2 <= decision.total:
            followup = [
                prompt[0],
                {"role": "user", "content": prompt[1]["content"] + "\n\n" + feedback(candidates)},
            ]
            refined = await self._round(followup, self.refine_samples, problem, evidence)
            weighted = [c for c, _ in refined for _ in range(2)]
            decision = decide([*weighted, *candidates])
            decision = replace(decision, total=len(candidates) + len(refined))
        return Solution(
            decision,
            tuple(sample for _, sample in first),
            tuple(e.id for e in shots),
            tuple(sample for _, sample in refined),
        )
