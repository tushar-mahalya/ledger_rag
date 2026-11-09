"""Program-of-thought reasoning core: DSL, grounding, voting, exemplars, solver."""

import asyncio
import hashlib
from pathlib import Path

import pytest

from ledger_rag.evaluation.official import finqa as official
from ledger_rag.reasoning.consistency import Candidate, decide, vote_key
from ledger_rag.reasoning.dsl import ProgramParseError, parse
from ledger_rag.reasoning.exemplars import Exemplar, ExemplarIndex
from ledger_rag.reasoning.grounding import Evidence, ungrounded
from ledger_rag.reasoning.solver import Execution, Problem, ProgramTask, Reply, Solver, messages

TABLE = [
    ["", "2009", "2008"],
    ["net revenue", "$ 5829", "$ 5735"],
    ["operating income", "( 120 )", "98"],
]
VENDORED = Path(__file__).parents[2] / "src/ledger_rag/evaluation/official/_finqa_upstream.py"


def test_vendored_official_executor_is_unmodified() -> None:
    digest = hashlib.sha256(VENDORED.read_bytes()).hexdigest()
    assert digest == "df28bcec6930856322d7cd391d9b85e6b0f11dd4484f7d1edb1540540facc9bc"


@pytest.mark.parametrize(
    ("reply", "program"),
    [
        ("subtract(5829, 5735), divide(#0, 5735)", "subtract(5829, 5735), divide(#0, 5735)"),
        (
            "FACTS:\n- revenue 2009: $5,829\nPROGRAM: subtract($5,829, $5,735), divide(#0, 5735)",
            "subtract(5829, 5735), divide(#0, 5735)",
        ),
        ("PROGRAM: ```divide(3.5%, const_100)```", "divide(3.5%, const_100)"),
        ("**Program:** add(.5, -120)", "add(.5, -120)"),
        ("PROGRAM: table_average(net revenue, none)", "table_average(net revenue, none)"),
        ("I add(1, 2) first.\n\nsubtract(5829, 5735)", "subtract(5829, 5735)"),
    ],
)
def test_parse_recovers_canonical_programs(reply: str, program: str) -> None:
    assert parse(reply).text == program


@pytest.mark.parametrize(
    "reply",
    [
        "the answer is 12",
        "PROGRAM: divide(#1, 5)",
        "PROGRAM: subtract(1, 2, 3)",
        "PROGRAM: add(revenue, 2)",
        "PROGRAM: table_sum(#0, none)",
        "PROGRAM: " + ", ".join("add(1, 2)" for _ in range(11)),
    ],
)
def test_parse_rejects_malformed_programs(reply: str) -> None:
    with pytest.raises(ProgramParseError):
        parse(reply)


def test_official_execution_semantics() -> None:
    assert official.execute("subtract(5829, 5735), divide(#0, 5735)", TABLE).value == round(
        94 / 5735, 5
    )
    assert official.execute("table_sum(net revenue, none)", TABLE).value == 11564.0
    assert official.execute("greater(5829, 5735)", TABLE).value == "yes"
    assert official.execute("divide(3.5%, const_100)", TABLE).value == 0.00035
    # Parenthesized cells are invalid in the official row parser; so is division by zero.
    assert not official.execute("table_sum(operating income, none)", TABLE).valid
    assert not official.execute("divide(1, 0)", TABLE).valid
    assert not official.execute("table_sum(missing row, none)", TABLE).valid


def test_official_and_tolerant_comparison() -> None:
    assert official.official_correct(0.01639, 0.01639)
    assert not official.official_correct(0.0164, 0.01639)
    assert official.tolerant_correct(0.0164, 0.01639)  # Rounded to 4 places.
    assert not official.tolerant_correct(1.64, 0.01639)
    assert official.tolerant_correct("yes", "yes")
    assert not official.official_correct(None, 0.5)


def test_grounding_accepts_evidence_numbers_and_rejects_invented_ones() -> None:
    evidence = Evidence.build(
        ["revenue grew to $1.2 billion over three years", "margin was 12.5%"],
        TABLE,
        "what is the change in net revenue?",
    )
    assert ungrounded(parse("subtract(5829, 5735), divide(#0, 5735)"), evidence) == []
    assert ungrounded(parse("multiply(1.2, 1000000000)"), evidence) == ["1000000000"]
    assert ungrounded(parse("divide(1200000000, 3)"), evidence) == []
    assert ungrounded(parse("multiply(12.5%, 120)"), evidence) == []
    assert ungrounded(parse("subtract(5829, 5000)"), evidence) == ["5000"]
    assert ungrounded(parse("table_max(gross profit, none)"), evidence) == ["gross profit"]
    assert ungrounded(parse("divide(5829, const_1000)"), evidence) == []


def test_vote_groups_equal_values_and_prefers_the_first_on_ties() -> None:
    assert vote_key(0.123456) == vote_key(0.1234561)
    decision = decide(
        [
            Candidate("a", 0.5),
            Candidate("b", 0.25),
            Candidate(None, None, "parse"),
            Candidate("c", 0.25),
            Candidate("d", 0.5000001),
        ]
    )
    assert decision.program == "a" and decision.votes == 2 and decision.valid == 4
    assert decision.agreement == pytest.approx(0.4)
    empty = decide([Candidate(None, None, "parse")])
    assert empty.value is None and empty.agreement == 0


def test_exemplar_search_ranks_similar_questions_and_excludes_same_page() -> None:
    index = ExemplarIndex(
        [
            Exemplar(
                "1",
                "what was the percentage change in net revenue from 2008 to 2009?",
                (),
                "p1",
                "A/2009/page_1.pdf",
            ),
            Exemplar("2", "what portion of total debt is due within one year?", (), "p2", "B"),
            Exemplar(
                "3", "what is the percentage change in net income from 2010 to 2011?", (), "p3", "C"
            ),
        ]
    )
    found = index.search("what is the percentage change in net revenue in 2012?", 2)
    assert [e.id for e in found] == ["1", "3"]
    excluded = index.search(
        "what is the percentage change in net revenue?", 1, exclude_group="A/2009/page_1.pdf"
    )
    assert [e.id for e in excluded] == ["3"]
    assert index.search("anything", 0) == []


class _Scripted:
    def __init__(self, replies: list[Reply]) -> None:
        self.replies = replies
        self.seen: list[int] = []

    async def read(self, prompt: list[dict[str, str]], sample: int) -> Reply:
        self.seen.append(sample)
        return self.replies[sample]


def _execute(program: object, problem: Problem) -> Execution:
    result = official.execute(program.text, problem.table)  # type: ignore[attr-defined]
    return Execution(result.value, result.valid)


def test_solver_votes_over_grounded_executed_samples() -> None:
    problem = Problem("q1", "what is the change in net revenue?", ["Revenue rose."], [], TABLE)
    reader = _Scripted(
        [
            Reply("PROGRAM: subtract(5829, 5000)"),  # Invented operand: no vote.
            Reply("PROGRAM: subtract(5829, 5735)"),
            Reply("FACTS:\n- x\nPROGRAM: subtract($5,829, 5735)"),
            Reply("", error="http_500"),
            Reply("no program here"),
        ]
    )
    solution = asyncio.run(
        Solver(reader=reader, task=ProgramTask(_execute), samples=5).solve(problem)
    )
    assert solution.decision.value == 94.0 and solution.decision.votes == 2
    assert [s.reason for s in solution.samples] == [
        "ungrounded",
        "ok",
        "ok",
        "provider",
        "parse",
    ]
    record = solution.record()
    assert record["sample_values"] == [None, 94.0, 94.0, None, None]
    assert "question" not in record and sorted(reader.seen) == [0, 1, 2, 3, 4]


def test_prompt_contains_exemplars_page_and_question_only_as_data() -> None:
    problem = Problem("q", "what is x?", ["Pre."], ["Post."], TABLE)
    shot = Exemplar("e", "what is y?", ("y is 2",), "add(1, 1)")
    rendered = messages(problem, [shot])
    assert rendered[0]["role"] == "system" and "PROGRAM:" in rendered[0]["content"]
    user = rendered[1]["content"]
    assert user.index("Solved examples") < user.index("Page:") < user.rindex("Question: what")
    assert "| net revenue | $ 5829 | $ 5735 |" in user
    with pytest.raises(ValueError):
        Solver(reader=_Scripted([]), task=ProgramTask(_execute), samples=0)
