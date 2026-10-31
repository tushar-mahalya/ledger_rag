import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from ledger_rag.config import Settings
from ledger_rag.providers import (
    AnthropicProvider,
    ExtractiveProvider,
    OpenAIProvider,
    ProviderError,
    make_provider,
)


def settings(provider: str = "openai", base: str = "https://custom.test/v1") -> Settings:
    return Settings(
        demo_token=SecretStr("demo-test-token"),
        provider=provider,  # type: ignore[arg-type]
        llm_model="test-model",
        llm_api_key=SecretStr("secret-key"),
        llm_base_url=base,
    )


async def transport(
    provider: OpenAIProvider | AnthropicProvider,
    response: Any,
    *,
    status: int = 200,
) -> list[httpx.Request]:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, json=response)

    await provider._client.aclose()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    return requests


async def test_openai_normalizes_tools_and_preserves_history() -> None:
    provider = OpenAIProvider(settings())
    requests = await transport(
        provider,
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": '{"term":"cash"}'},
                            }
                        ],
                    }
                }
            ]
        },
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "previous", "name": "lookup", "arguments": {"term": "debt"}}],
        },
        {"role": "tool", "tool_call_id": "previous", "content": '{"value":10}'},
        {"role": "user", "content": "cash?"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object"},
            },
        }
    ]
    result = await provider.complete(messages, tools)
    assert result == {
        "content": "",
        "tool_calls": [{"id": "call-1", "name": "lookup", "arguments": {"term": "cash"}}],
    }
    body = json.loads(requests[0].content)
    assert str(requests[0].url) == "https://custom.test/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer secret-key"
    assert body["tools"] == tools
    assert body["messages"][0]["tool_calls"][0]["function"]["arguments"] == '{"term": "debt"}'
    assert body["messages"][1] == messages[1]
    assert messages[0]["tool_calls"][0]["arguments"] == {"term": "debt"}
    await provider.close()
    assert provider._client.is_closed


@pytest.mark.parametrize(
    "base",
    [
        "https://custom.test/v1/",
        "https://custom.test/v1/messages",
    ],
)
async def test_anthropic_converts_parallel_tools_and_results(base: str) -> None:
    provider = AnthropicProvider(settings("anthropic", base))
    requests = await transport(
        provider,
        {
            "content": [
                {"type": "text", "text": "Evidence [C1]"},
                {"type": "tool_use", "id": "new", "name": "lookup", "input": {"term": "cash"}},
            ]
        },
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "Grounded only."},
        {"role": "user", "content": "Question"},
        {
            "role": "assistant",
            "content": "Checking",
            "tool_calls": [
                {"id": "one", "name": "lookup", "arguments": {"term": "one"}},
                {"id": "two", "function": {"name": "lookup", "arguments": '{"term":"two"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "one", "content": "{}"},
        {"role": "tool", "tool_call_id": "two", "content": "{}"},
    ]
    result = await provider.complete(
        messages,
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Find evidence",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )
    assert result["tool_calls"] == [{"id": "new", "name": "lookup", "arguments": {"term": "cash"}}]
    body = json.loads(requests[0].content)
    assert str(requests[0].url) == "https://custom.test/v1/messages"
    assert requests[0].headers["x-api-key"] == "secret-key"
    assert requests[0].headers["anthropic-version"] == "2023-06-01"
    assert body["system"] == "Grounded only."
    assert len(body["messages"]) == 3
    assert body["messages"][1]["content"][2] == {
        "type": "tool_use",
        "id": "two",
        "name": "lookup",
        "input": {"term": "two"},
    }
    assert body["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "one", "content": "{}"},
        {"type": "tool_result", "tool_use_id": "two", "content": "{}"},
    ]
    assert body["tools"][0]["input_schema"] == {"type": "object"}
    await provider.close()


@pytest.mark.parametrize("provider_type", [OpenAIProvider, AnthropicProvider])
@pytest.mark.parametrize("status", [401, 429, 500])
async def test_http_errors_are_sanitized_and_not_retried(
    provider_type: type[OpenAIProvider] | type[AnthropicProvider], status: int
) -> None:
    provider = provider_type(settings())
    requests = await transport(provider, {"error": "secret-key PROMPT"}, status=status)
    with pytest.raises(ProviderError) as error:
        await provider.complete([{"role": "user", "content": "PROMPT"}])
    assert str(status) in str(error.value)
    assert "secret-key" not in str(error.value) and "PROMPT" not in str(error.value)
    assert len(requests) == 1
    await provider.close()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"content": 1}}]},
        {"choices": [{"message": {"content": None}}]},
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"id": "one", "function": {"name": "lookup", "arguments": "[]"}}
                        ],
                    }
                }
            ]
        },
    ],
)
async def test_openai_invalid_response(payload: Any) -> None:
    provider = OpenAIProvider(settings())
    await transport(provider, payload)
    with pytest.raises(ProviderError, match="Invalid OpenAI"):
        await provider.complete([])
    await provider.close()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"content": []},
        {"content": [{"type": "text", "text": None}]},
        {"content": [{"type": "tool_use", "id": "one", "name": "lookup", "input": []}]},
    ],
)
async def test_anthropic_invalid_response(payload: Any) -> None:
    provider = AnthropicProvider(settings())
    await transport(provider, payload)
    with pytest.raises(ProviderError, match="Invalid Anthropic"):
        await provider.complete([])
    await provider.close()


async def test_timeout_and_invalid_json() -> None:
    provider = OpenAIProvider(settings())
    assert provider._client.timeout.read == 20
    assert provider._client.timeout.connect == 5
    await provider._client.aclose()

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("secret-key PROMPT", request=request)

    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(timeout))
    with pytest.raises(ProviderError, match="transport failure"):
        await provider.complete([])
    await provider.close()
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"not json"))
    )
    with pytest.raises(ProviderError, match="invalid JSON"):
        await provider.complete([])
    await provider.close()


async def test_extractive_quotes_evidence_or_abstains() -> None:
    provider = ExtractiveProvider()
    message = {
        "role": "user",
        "content": json.dumps(
            {
                "question": "What was revenue?",
                "profile": "voice",
                "evidence": [{"id": "C1", "text": "Revenue was $100. Debt was $20."}],
            }
        ),
    }
    assert await provider.complete([{"role": "system", "content": "Grounded."}, message]) == {
        "content": "Revenue was $100. [C1]",
        "tool_calls": [],
    }
    message["content"] = json.dumps(
        {
            "question": "What are dividends?",
            "evidence": [{"id": "C1", "text": "Revenue was $100."}],
        }
    )
    assert (await provider.complete([message]))["content"] == ""
    assert (await provider.complete([{"role": "user", "content": "guess"}]))["content"] == ""
    await provider.close()


async def test_factory() -> None:
    for name, cls in [
        ("extractive", ExtractiveProvider),
        ("openai", OpenAIProvider),
        ("anthropic", AnthropicProvider),
    ]:
        provider = make_provider(settings(name))
        assert isinstance(provider, cls)
        await provider.close()


@pytest.mark.parametrize("provider_type", [OpenAIProvider, AnthropicProvider])
async def test_search_evidence_tool_round_trip(
    provider_type: type[OpenAIProvider] | type[AnthropicProvider],
) -> None:
    provider = provider_type(settings())
    response = (
        {"choices": [{"message": {"content": "Revenue increased. [C1]"}}]}
        if provider_type is OpenAIProvider
        else {"content": [{"type": "text", "text": "Revenue increased. [C1]"}]}
    )
    requests = await transport(provider, response)
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "What was revenue?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "search-1",
                    "name": "search_evidence",
                    "arguments": {"question": "revenue"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "search_evidence",
            "tool_call_id": "search-1",
            "content": '{"evidence":[{"id":"C1","text":"Revenue increased."}]}',
        },
    ]
    result = await provider.complete(
        history,
        [
            {
                "type": "function",
                "function": {
                    "name": "search_evidence",
                    "parameters": {
                        "type": "object",
                        "properties": {"question": {"type": "string"}},
                        "required": ["question"],
                    },
                },
            }
        ],
    )
    assert result == {"content": "Revenue increased. [C1]", "tool_calls": []}
    body = json.loads(requests[0].content)
    if provider_type is OpenAIProvider:
        assert body["messages"][1]["tool_calls"][0]["function"]["name"] == "search_evidence"
        assert body["messages"][2]["name"] == "search_evidence"
    else:
        assert body["messages"][1]["content"][0]["name"] == "search_evidence"
        assert body["messages"][1]["content"][0]["input"] == {"question": "revenue"}
        assert body["messages"][2]["content"][0]["tool_use_id"] == "search-1"
    await provider.close()
