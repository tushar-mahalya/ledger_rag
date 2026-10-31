import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from ledger_rag.core.config import Settings
from ledger_rag.generation.providers import (
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


@pytest.mark.parametrize(
    "question",
    [
        "What was revenue in 2025?",
        "What was FY2025 revenue?",
        "What was fiscal year 2025 revenue?",
    ],
)
async def test_extractive_single_year_excludes_other_periods(question: str) -> None:
    provider = ExtractiveProvider()
    result = await provider.complete(
        [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "evidence": [
                            {"id": "C1", "text": "Revenue in 2024 was $100."},
                            {"id": "C2", "text": "Revenue in 2025 was $120."},
                            {
                                "id": "C3",
                                "text": "Revenue was $80.",
                                "title": "2023 Results",
                                "published_at": "2025-01-01",
                            },
                        ],
                    }
                ),
            }
        ]
    )
    assert result["content"] == "Revenue in 2025 was $120. [C2]"
    await provider.close()


@pytest.mark.parametrize(
    "evidence",
    [
        [{"id": "C1", "text": "Revenue was $100."}],
        [{"id": "C1", "text": "Revenue was $100.", "published_at": "2025-01-01"}],
        [{"id": "C1", "text": "Revenue was $100.", "title": "2024 and 2025 Results"}],
        [{"id": "C1", "text": "Revenue was $100. Debt in 2024 was $20.", "title": "2025 Results"}],
        [{"id": "C1", "text": "Revenue was $100. In 2025 management announced buybacks."}],
        [
            {
                "id": "C1",
                "text": "Revenue in 2025 was $100. Revenue in 2024 was $90.",
                "title": "2025 Results",
            }
        ],
    ],
)
async def test_extractive_single_year_handles_ambiguous_context(evidence: Any) -> None:
    provider = ExtractiveProvider()
    result = await provider.complete(
        [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": "What was revenue in 2025?",
                        "evidence": evidence,
                    }
                ),
            }
        ]
    )
    if "Revenue in 2025" in evidence[0]["text"]:
        assert result["content"] == "Revenue in 2025 was $100. [C1]"
    elif evidence[0].get("title") == "2025 Results":
        assert result["content"] == "Revenue was $100. [C1]"
    else:
        assert result["content"] == ""
    await provider.close()


async def test_extractive_unambiguous_title_preserves_multiple_document_evidence() -> None:
    provider = ExtractiveProvider()
    result = await provider.complete(
        [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": "What was Acme and Beta revenue in 2025?",
                        "evidence": [
                            {
                                "id": "C1",
                                "title": "Acme FY2025 Results",
                                "text": "Acme revenue was $100.",
                            },
                            {
                                "id": "C2",
                                "title": "Beta 2025 Results",
                                "text": "Beta revenue was $200.",
                            },
                            {
                                "id": "C3",
                                "title": "Acme 2024 Results",
                                "text": "Acme revenue was $90.",
                            },
                        ],
                    }
                ),
            }
        ]
    )
    assert result["content"] == "Acme revenue was $100. [C1]\n\nBeta revenue was $200. [C2]"
    await provider.close()


@pytest.mark.parametrize(
    "question",
    [
        "Compare revenue in 2024 and 2025.",
        "How did revenue change in 2025 versus the prior year?",
        "What was revenue growth in 2025?",
        "What revenue was reported as of 2025?",
        "What was revenue before 2025?",
    ],
)
async def test_extractive_comparisons_and_ranges_preserve_lexical_evidence(question: str) -> None:
    provider = ExtractiveProvider()
    result = await provider.complete(
        [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "evidence": [
                            {"id": "C1", "text": "Revenue in 2024 was $100."},
                            {"id": "C2", "text": "Revenue in 2025 was $120."},
                        ],
                    }
                ),
            }
        ]
    )
    assert "[C1]" in result["content"] and "[C2]" in result["content"]
    await provider.close()


async def test_extractive_year_overlap_alone_does_not_answer_unrelated_topic() -> None:
    provider = ExtractiveProvider()
    result = await provider.complete(
        [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": "What were dividends in 2025?",
                        "evidence": [{"id": "C1", "text": "Revenue in 2025 was $120."}],
                    }
                ),
            }
        ]
    )
    assert result["content"] == ""
    await provider.close()


@pytest.mark.parametrize("year", [1998, 2021, 2028])
async def test_extractive_period_rule_is_not_specific_to_one_year(year: int) -> None:
    provider = ExtractiveProvider()
    result = await provider.complete(
        [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": f"What was revenue in {year}?",
                        "evidence": [
                            {"id": "C1", "text": f"Revenue in {year - 1} was $100."},
                            {"id": "C2", "text": f"Revenue in {year} was $120."},
                        ],
                    }
                ),
            }
        ]
    )
    assert result["content"] == f"Revenue in {year} was $120. [C2]"
    await provider.close()


async def test_extractive_mixed_year_sentence_preserves_full_restatement_attribution() -> None:
    provider = ExtractiveProvider()
    evidence = [
        {
            "id": "C1",
            "text": "In 2025 the company restated 2024 revenue to $100.",
        }
    ]
    result = await provider.complete(
        [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": "What was revenue in 2025?",
                        "evidence": evidence,
                    }
                ),
            }
        ]
    )
    assert result["content"] == "In 2025 the company restated 2024 revenue to $100. [C1]"
    result = await provider.complete(
        [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": "Compare revenue reporting in 2024 and 2025.",
                        "evidence": evidence,
                    }
                ),
            }
        ]
    )
    assert result["content"] == "In 2025 the company restated 2024 revenue to $100. [C1]"
    await provider.close()


@pytest.mark.parametrize("separator", [", compared with ", " compared to ", "; versus "])
async def test_extractive_quotes_full_target_period_comparative_sentence(
    separator: str,
) -> None:
    provider = ExtractiveProvider()
    text = (
        "In fiscal year 2024 Acme revenue was USD 120 million"
        + separator
        + "revenue of USD 100 million in fiscal year 2023."
    )
    evidence = [
        {"id": "C1", "text": text},
        {"id": "C2", "text": "Revenue in fiscal year 2023 was USD 100 million."},
    ]
    result = await provider.complete(
        [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": "What was Acme revenue in fiscal year 2024?",
                        "evidence": evidence,
                    }
                ),
            }
        ]
    )
    assert result["content"] == f"{text} [C1]"
    comparison = await provider.complete(
        [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": "Compare revenue in fiscal years 2023 and 2024.",
                        "evidence": evidence[:1],
                    }
                ),
            }
        ]
    )
    assert comparison["content"] == f"{text} [C1]"
    await provider.close()


async def test_extractive_does_not_remove_prior_period_attribution_after_year_heading() -> None:
    provider = ExtractiveProvider()
    result = await provider.complete(
        [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": "What was revenue in 2025?",
                        "evidence": [
                            {
                                "id": "C1",
                                "text": "In 2025, the company restated revenue in 2024 to $100.",
                            }
                        ],
                    }
                ),
            }
        ]
    )
    assert result["content"] == ("In 2025, the company restated revenue in 2024 to $100. [C1]")
    await provider.close()


async def test_extractive_comparative_quote_preserves_numeric_thousands_separator() -> None:
    provider = ExtractiveProvider()
    result = await provider.complete(
        [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": "What was revenue in 2025?",
                        "evidence": [
                            {
                                "id": "C1",
                                "text": "Revenue in 2025 was $1,200, compared with $900 in 2024.",
                            }
                        ],
                    }
                ),
            }
        ]
    )
    assert result["content"] == "Revenue in 2025 was $1,200, compared with $900 in 2024. [C1]"
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
