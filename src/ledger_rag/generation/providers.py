"""Small HTTP adapters with an OpenAI-normalized tool contract.

Transient failures (429/5xx overload, transport errors) are retried within the
request deadline by ``core.retry``; other statuses fail fast and sanitized.
"""

import json
import re
from typing import Any

import httpx

from ledger_rag.core.config import Settings
from ledger_rag.core.contracts import Provider
from ledger_rag.core.retry import RetryPolicy, send_with_retries


class ProviderError(RuntimeError):
    """A sanitized upstream transport or protocol failure."""


def _endpoint(base: str, suffix: str) -> str:
    base = base.rstrip("/")
    return base if base.endswith("/" + suffix) else base + "/" + suffix


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("Tool arguments must be an object")
    return value


def _call(value: dict[str, Any]) -> dict[str, Any]:
    function = value.get("function", value)
    if not isinstance(function, dict):
        raise ValueError("Invalid function")
    call_id, name = value.get("id"), function.get("name")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("Missing tool id")
    if not isinstance(name, str) or not name:
        raise ValueError("Missing tool name")
    return {"id": call_id, "name": name, "arguments": _arguments(function.get("arguments"))}


def _years(text: str) -> set[str]:
    return set(re.findall(r"\b(?:FY\s*)?((?:19|20)\d{2})\b", text, flags=re.IGNORECASE))


def _single_period(question: str) -> str | None:
    years = _years(question)
    if len(years) != 1 or re.search(
        r"\b(?:compar\w*|versus|vs|between|before|after|since|through|until|"
        r"trend\w*|growth|chang\w*|previous|prior)\b|"
        r"\bas\s+of\b|\byear[\s-]+over[\s-]+year\b|\blast\s+year\b",
        question,
        flags=re.IGNORECASE,
    ):
        return None
    return next(iter(years))


class ExtractiveProvider:
    """Lexical demo, not semantic reasoning.

    Single explicit-year questions require that year in a quoted sentence,
    or an unambiguous document-title year for sentences without explicit years.
    Comparative sentences retain their complete period/value attribution.
    Publication dates never establish the financial reporting period.
    Comparisons and temporal ranges retain the ordinary lexical selection.
    """

    model = "extractive-demo"

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        del tools
        content = ""
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            try:
                data = json.loads(message.get("content", ""))
            except (ValueError, TypeError):
                break
            if not isinstance(data, dict):
                break
            question = str(data.get("question", ""))
            requested_year = _single_period(question)
            question_years = _years(question)
            words = {
                word
                for word in re.findall(r"\w+", question.lower())
                if len(word) > 2
                and word not in question_years
                and not re.fullmatch(r"fy(?:19|20)\d{2}", word)
                and word
                not in {
                    "the",
                    "and",
                    "was",
                    "were",
                    "what",
                    "which",
                    "how",
                    "for",
                    "with",
                    "does",
                    "did",
                    "are",
                    "that",
                    "from",
                    "this",
                    "about",
                    "fiscal",
                    "year",
                }
            }
            candidates: list[tuple[int, str]] = []
            evidence = data.get("evidence", [])
            if not isinstance(evidence, list):
                break
            for item in evidence:
                if not isinstance(item, dict):
                    continue
                marker, text = item.get("id"), item.get("text")
                if not isinstance(marker, str) or not re.fullmatch(r"C[1-9]\d*", marker):
                    continue
                if not isinstance(text, str):
                    continue
                title = item.get("title", "")
                title_years = _years(title) if isinstance(title, str) else set()
                for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
                    if requested_year:
                        sentence_years = _years(sentence)
                        if sentence_years:
                            if requested_year not in sentence_years:
                                continue
                        elif title_years != {requested_year}:
                            continue
                    overlap = len(words & set(re.findall(r"\w+", sentence.lower())))
                    if overlap and sentence.strip():
                        candidates.append((overlap, f"{sentence.strip()} [{marker}]"))
            candidates.sort(key=lambda item: item[0], reverse=True)
            count = 1 if data.get("profile") == "voice" else 3
            content = "\n\n".join(dict.fromkeys(item[1] for item in candidates[:count]))
            break
        return {"content": content, "tool_calls": []}

    async def close(self) -> None:
        return None


class _HTTPProvider:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.llm_model
        self._base_url = settings.llm_base_url
        self._key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else ""
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens
        self._retry = RetryPolicy.from_settings(settings)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=5.0),
            follow_redirects=False,
        )

    async def _post(self, suffix: str, body: dict[str, Any], headers: dict[str, str]) -> Any:
        try:
            request = self._client.build_request(
                "POST", _endpoint(self._base_url, suffix), json=body, headers=headers
            )
            response = await send_with_retries(
                lambda: self._client.send(request), self._retry, operation="llm"
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"LLM endpoint returned HTTP {exc.response.status_code}") from None
        except httpx.HTTPError:
            raise ProviderError("LLM endpoint transport failure") from None
        try:
            return response.json()
        except ValueError:
            raise ProviderError("LLM endpoint returned invalid JSON") from None

    async def close(self) -> None:
        await self._client.aclose()


class OpenAIProvider(_HTTPProvider):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        # Only explicitly configured controls are sent; compatible servers differ on
        # which they accept (e.g. newer OpenAI models need max_completion_tokens).
        self._generation: dict[str, Any] = {
            key: value
            for key, value in {
                "temperature": settings.llm_temperature,
                settings.llm_max_tokens_parameter: settings.llm_max_tokens,
                "seed": settings.llm_seed,
                "reasoning_effort": settings.llm_reasoning_effort,
            }.items()
            if value is not None
        }

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        wire_messages = []
        try:
            for message in messages:
                wire = dict(message)
                if "tool_calls" in wire:
                    wire["tool_calls"] = [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"]),
                            },
                        }
                        for call in (_call(value) for value in wire["tool_calls"])
                    ]
                wire_messages.append(wire)
        except (ValueError, TypeError, KeyError, AttributeError):
            raise ProviderError("Invalid tool-call history") from None
        body: dict[str, Any] = {
            "model": self.model,
            "messages": wire_messages,
            **self._generation,
        }
        if tools:
            body["tools"] = tools
        data = await self._post("chat/completions", body, {"Authorization": f"Bearer {self._key}"})
        try:
            choices = data["choices"]
            if not isinstance(choices, list) or not choices:
                raise ValueError("Missing choices")
            message = choices[0]["message"]
            content = message.get("content")
            calls = message.get("tool_calls", [])
            if content is None and calls:
                content = ""
            if not isinstance(content, str) or not isinstance(calls, list):
                raise ValueError("Invalid content")
            return {"content": content, "tool_calls": [_call(call) for call in calls]}
        except (ValueError, TypeError, KeyError, AttributeError, IndexError):
            raise ProviderError("Invalid OpenAI completion response") from None


class AnthropicProvider(_HTTPProvider):
    DEFAULT_MAX_TOKENS = 2048

    def __init__(self, settings: Settings) -> None:
        # Fail at startup rather than silently ignoring controls Messages lacks.
        if settings.llm_seed is not None or settings.llm_reasoning_effort is not None:
            raise ProviderError("Anthropic does not support llm_seed or llm_reasoning_effort")
        if settings.llm_temperature is not None and settings.llm_temperature > 1:
            raise ProviderError("Anthropic llm_temperature must be between 0 and 1")
        super().__init__(settings)

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        system: list[str] = []
        wire_messages: list[dict[str, Any]] = []
        try:
            for message in messages:
                role = message["role"]
                content = message.get("content") or ""
                if not isinstance(content, str):
                    raise ValueError("Text history required")
                if role == "system":
                    system.append(content)
                    continue
                blocks: list[dict[str, Any]] = []
                if role == "tool":
                    role = "user"
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": message["tool_call_id"],
                            "content": content,
                        }
                    )
                elif role in {"assistant", "user"}:
                    if content:
                        blocks.append({"type": "text", "text": content})
                    for value in message.get("tool_calls", []):
                        call = _call(value)
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": call["id"],
                                "name": call["name"],
                                "input": call["arguments"],
                            }
                        )
                else:
                    raise ValueError("Invalid role")
                if wire_messages and wire_messages[-1]["role"] == role:
                    wire_messages[-1]["content"].extend(blocks)
                else:
                    wire_messages.append({"role": role, "content": blocks})
            body: dict[str, Any] = {
                "model": self.model,
                "max_tokens": self._max_tokens or self.DEFAULT_MAX_TOKENS,
                "messages": wire_messages,
            }
            if self._temperature is not None:
                body["temperature"] = self._temperature
            if system:
                body["system"] = "\n\n".join(system)
            if tools:
                body["tools"] = [
                    {
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "input_schema": tool["function"]["parameters"],
                    }
                    for tool in tools
                ]
        except (ValueError, TypeError, KeyError, AttributeError):
            raise ProviderError("Invalid Anthropic tool or message history") from None
        data = await self._post(
            "messages",
            body,
            {
                "x-api-key": self._key,
                "Authorization": f"Bearer {self._key}",
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            blocks = data["content"]
            if not isinstance(blocks, list) or not blocks:
                raise ValueError("Missing content")
            texts: list[str] = []
            calls: list[dict[str, Any]] = []
            for block in blocks:
                if block["type"] == "text":
                    if not isinstance(block["text"], str):
                        raise ValueError("Invalid text")
                    texts.append(block["text"])
                elif block["type"] == "tool_use":
                    calls.append(
                        _call(
                            {
                                "id": block["id"],
                                "name": block["name"],
                                "arguments": block["input"],
                            }
                        )
                    )
                elif block["type"] not in {"thinking", "redacted_thinking"}:
                    raise ValueError("Unknown content block")
            if not texts and not calls:
                raise ValueError("No completion")
            return {"content": "\n".join(texts), "tool_calls": calls}
        except (ValueError, TypeError, KeyError, AttributeError):
            raise ProviderError("Invalid Anthropic completion response") from None


def make_provider(settings: Settings) -> Provider:
    if settings.provider == "extractive":
        return ExtractiveProvider()
    if settings.provider == "openai":
        return OpenAIProvider(settings)
    return AnthropicProvider(settings)
