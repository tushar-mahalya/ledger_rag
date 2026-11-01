import importlib.util
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "bounded_eval_session", ROOT / "scripts/run_eval_session.py"
)
assert spec is not None and spec.loader is not None
session = importlib.util.module_from_spec(spec)
spec.loader.exec_module(session)


@pytest.mark.parametrize(
    "seconds,spent", [(9001, 0), (0, 0), (4500, 20), (1, -1), (1, float("nan"))]
)
def test_admission_fails_closed(seconds: int, spent: float) -> None:
    with pytest.raises(ValueError):
        session.admission(seconds, spent)


def test_budget_is_total_not_credit_balance() -> None:
    plan = session.admission(9000, 0)
    assert plan["reserved_usd"] == pytest.approx(24.9811)
    assert plan["remaining_contingency_usd"] == pytest.approx(3.0189)
    assert plan["calls"] == 1200 and not plan["test_unsealed"]
    assert len(session.ARMS) == 2
    assert session.ARMS[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert session.ARMS[1]["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "low"


@pytest.mark.parametrize("failure", [None, TimeoutError(), KeyboardInterrupt(), RuntimeError()])
def test_remote_call_always_terminates_container(failure: BaseException | None) -> None:
    events: list[Any] = []

    class Call:
        def get(self, *, timeout: float) -> str:
            events.append(("get", timeout))
            if failure is not None:
                raise failure
            return "committed"

        def cancel(self, *, terminate_containers: bool) -> None:
            events.append(("cancel", terminate_containers))

    class Function:
        def with_options(self, **options: Any) -> "Function":
            assert options["cpu"] == (2, 2)
            assert options["memory"] == (8192, 8192)
            assert options["max_containers"] == 1 and options["retries"] == 0
            assert options["timeout"] == 1
            return self

        def spawn(self, *args: Any) -> Call:
            events.append(("spawn", args))
            return Call()

    deadline = session.time.monotonic() + 1
    if failure is None:
        assert session.bounded_call(Function(), deadline, "options") == "committed"
    else:
        with pytest.raises(type(failure)):
            session.bounded_call(Function(), deadline, "options")
    assert events[-1] == ("cancel", True)
    assert 0 < events[1][1] <= 1


def test_expired_session_never_submits() -> None:
    with pytest.raises(TimeoutError, match="before submission"):
        session.bounded_call(None, session.time.monotonic() - 1)


def test_ledger_reserves_all_runs_without_refunding_early_exits() -> None:
    plan = session.admission(4500, 0)
    first = session.reserve_budget({}, plan)
    second = session.reserve_budget(first, plan)
    assert second["reserved_seconds"] == 9000
    assert second["reserved_usd"] == pytest.approx(24.9812)
    with pytest.raises(ValueError, match="cumulative"):
        session.reserve_budget(second, session.admission(1, 0))
    with pytest.raises(ValueError, match="invalid"):
        session.reserve_budget(
            {"reserved_seconds": -1, "reserved_usd": 0, "prior_session_spend_usd": 0}, plan
        )


def test_best_two_are_exact_pairs_not_an_accidental_cartesian_product(tmp_path: Path) -> None:
    selection = tmp_path / "pairs.json"
    selection.write_text(
        '[{"label":"qwen:off","mode":"retrieved@5"},{"label":"qwen:low","mode":"oracle"}]'
    )
    assert len(session.selected_pairs(selection)) == 2
    selection.write_text('[{"label":"qwen:off","mode":"test"}]')
    with pytest.raises(ValueError, match="exactly two"):
        session.selected_pairs(selection)
