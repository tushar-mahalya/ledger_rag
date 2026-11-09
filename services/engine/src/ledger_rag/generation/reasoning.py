"""Separate model reasoning from answer text for OpenAI-compatible responses.

Reasoning servers (for example vLLM with ``--reasoning-parser``) return thinking in
``message.reasoning_content`` or ``message.reasoning``, not in ``content``. Without
a parser, or when a reply is truncated mid-thought, ``<think>`` text leaks into
``content``. Callers keep only the answer part and never parse the reasoning.
"""

import re
from collections.abc import Mapping
from typing import Any

REASONING_FIELDS = ("reasoning_content", "reasoning")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.I | re.S)
_THINK_OPEN = re.compile(r"<think>", re.I)
_THINK_CLOSE = re.compile(r"</think>", re.I)


def split_reasoning(text: str) -> tuple[str, str]:
    """Return ``(answer, reasoning)`` for text that may contain leaked thinking.

    Handles closed ``<think>`` blocks, a bare ``</think>`` whose opening tag was part
    of the prompt (templates that open the assistant turn with ``<think>``), and an
    unterminated ``<think>`` from a truncated reply: everything after it is reasoning.
    """
    reasoning: list[str] = []

    def keep(match: re.Match[str]) -> str:
        reasoning.append(match.group(0))
        return ""

    answer = _THINK_BLOCK.sub(keep, text)
    closes = list(_THINK_CLOSE.finditer(answer))
    if closes:
        reasoning.append(answer[: closes[-1].end()])
        answer = answer[closes[-1].end() :]
    opened = _THINK_OPEN.search(answer)
    if opened:
        reasoning.append(answer[opened.start() :])
        answer = answer[: opened.start()]
    return answer, "".join(reasoning)


def message_reasoning(message: Mapping[str, Any]) -> str:
    """Server-separated reasoning text from a message or stream delta, if any."""
    for name in REASONING_FIELDS:
        value = message.get(name)
        if isinstance(value, str) and value:
            return value
    return ""
