"""Subprocess boundary for operator tooling.

Commands receive secrets only through stdin or a child environment, never argv, and
errors never echo stdout (which may hold decrypted values).
"""

import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


class OpsError(RuntimeError):
    """A sanitized operator-tool failure; messages never contain secret values."""


@dataclass(frozen=True)
class Result:
    returncode: int
    stdout: bytes
    stderr: bytes


Runner = Callable[[Sequence[str], bytes | None, Mapping[str, str] | None, Path | None], Result]


def run(
    args: Sequence[str],
    stdin: bytes | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> Result:
    try:
        completed = subprocess.run(
            list(args),
            input=stdin,
            env=dict(env) if env is not None else None,
            cwd=cwd,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        raise OpsError(f"{args[0]} is not installed or not on PATH") from None
    return Result(completed.returncode, completed.stdout, completed.stderr)


def first_line(stderr: bytes, limit: int = 200) -> str:
    """A short diagnostic from stderr; callers use it only for tools that never print values."""
    lines = stderr.decode(errors="replace").strip().splitlines()
    return lines[-1][:limit] if lines else "no diagnostic output"


def check(result: Result, action: str, *, diagnostic: bool = True) -> Result:
    if result.returncode != 0:
        detail = f": {first_line(result.stderr)}" if diagnostic else ""
        raise OpsError(f"{action} failed (exit {result.returncode}){detail}")
    return result
