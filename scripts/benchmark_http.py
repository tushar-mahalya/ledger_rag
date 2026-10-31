"""Bounded closed-loop HTTP benchmark. Token is accepted only through the environment."""

import argparse
import asyncio
import codecs
import json
import math
import os
import re
import time
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from ledger_rag.core.models import Answer, Profile, QueryRequest
from ledger_rag.evaluation.harness import latency_summary

MAX_RESPONSE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class LoadConfig:
    url: str = "http://127.0.0.1:8000"
    question: str = "What was Aurora revenue in fiscal year 2024?"
    profile: Profile = Profile.CHAT
    top_k: int = 5
    concurrency: int = 4
    requests: int | None = 100
    duration: float | None = None
    warmup: int = 5
    timeout: float = 30
    stream: bool = False
    token_env: str = "LEDGER_BENCHMARK_TOKEN"

    def validate(self) -> None:
        parsed = urlsplit(self.url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Use an HTTP(S) base URL without credentials, query or fragment")
        if not 1 <= self.concurrency <= 1000 or not 0 <= self.warmup <= 10000:
            raise ValueError("Concurrency must be 1..1000; warmup 0..10000")
        if not math.isfinite(self.timeout) or not 0 < self.timeout <= 180:
            raise ValueError("Timeout must be finite in (0, 180]")
        if (self.requests is None) == (self.duration is None):
            raise ValueError("Specify exactly one of requests and duration")
        if self.requests is not None and not 1 <= self.requests <= 1_000_000:
            raise ValueError("Requests must be 1..1000000")
        if self.duration is not None and (
            not math.isfinite(self.duration) or not 0 < self.duration <= 3600
        ):
            raise ValueError("Duration must be finite in (0, 3600]")
        QueryRequest(question=self.question, profile=self.profile, top_k=self.top_k)


@dataclass
class Observation:
    elapsed_ms: float
    success: bool = False
    ttfe_ms: float | None = None
    validated_answer_ms: float | None = None
    error: str | None = None
    abstained: bool | None = None


def validate_answer(payload: Any, profile: Profile) -> Answer:
    answer = Answer.model_validate(payload)
    if answer.profile != profile or not answer.answer.strip():
        raise ValueError("Invalid answer profile or empty answer")
    if not answer.abstained and not answer.citations:
        raise ValueError("Non-abstaining answer requires citations")
    if answer.abstained and answer.citations:
        raise ValueError("Abstention must not have citations")
    if any(
        not math.isfinite(value) or value < 0
        for value in (answer.retrieval_ms, answer.generation_ms)
    ):
        raise ValueError("Invalid server timing")
    return answer


async def _bounded_sse_lines(response: httpx.Response) -> AsyncIterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8-sig")()
    buffer = ""
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > MAX_RESPONSE_BYTES:
            raise ValueError("Stream exceeds budget")
        buffer += decoder.decode(chunk)
        # Hold a trailing CR so a split CRLF never becomes two line endings.
        trailing_cr = buffer.endswith("\r")
        lines = re.split(r"\r\n|\r|\n", buffer[:-1] if trailing_cr else buffer)
        for line in lines[:-1]:
            yield line
        buffer = lines[-1] + ("\r" if trailing_cr else "")
    buffer += decoder.decode(b"", final=True)
    lines = re.split(r"\r\n|\r|\n", buffer)
    for line in lines[:-1]:
        yield line
    if lines[-1]:
        yield lines[-1]


async def query_once(client: httpx.AsyncClient, config: LoadConfig) -> Observation:
    started = time.perf_counter()
    observation = Observation(elapsed_ms=0)
    request = QueryRequest(question=config.question, profile=config.profile, top_k=config.top_k)
    endpoint = "/v1/query/stream" if config.stream else "/v1/query"
    try:
        # Absolute deadline bounds streams even when a peer keeps sending heartbeat bytes.
        async with asyncio.timeout(config.timeout):
            async with client.stream(
                "POST",
                config.url.rstrip("/") + endpoint,
                json=request.model_dump(mode="json"),
            ) as response:
                if response.status_code != 200:
                    observation.error = f"http_{response.status_code}"
                    return observation
                if not config.stream:
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                            raise ValueError("Response exceeds budget")
                        body.extend(chunk)
                    answer = validate_answer(json.loads(body), config.profile)
                    observation.validated_answer_ms = (time.perf_counter() - started) * 1000
                    observation.abstained = answer.abstained
                else:
                    if "text/event-stream" not in response.headers.get("content-type", ""):
                        raise ValueError("Expected SSE content type")
                    event = ""
                    data: list[str] = []
                    done = False
                    answer_seen = False
                    async for line in _bounded_sse_lines(response):
                        if line.startswith(":"):
                            continue
                        if line:
                            field, _, value = line.partition(":")
                            value = value.removeprefix(" ")
                            if field == "event":
                                event = value
                            elif field == "data":
                                data.append(value)
                            continue
                        if not data:
                            event = ""
                            continue
                        payload = json.loads("\n".join(data))
                        now_ms = (time.perf_counter() - started) * 1000
                        if event == "error":
                            observation.error = "sse_error"
                            return observation
                        if done:
                            raise ValueError("Event after done")
                        if event == "evidence":
                            if observation.ttfe_ms is not None or answer_seen:
                                raise ValueError("Out-of-order evidence")
                            if (
                                not isinstance(payload, dict)
                                or not isinstance(payload.get("sources"), list)
                                or not all(isinstance(item, str) for item in payload["sources"])
                                or not isinstance(payload.get("retrieval_ms"), (int, float))
                                or isinstance(payload["retrieval_ms"], bool)
                                or not math.isfinite(payload["retrieval_ms"])
                                or payload["retrieval_ms"] < 0
                            ):
                                raise ValueError("Invalid evidence event")
                            observation.ttfe_ms = now_ms
                        elif event == "answer":
                            if answer_seen or observation.ttfe_ms is None:
                                raise ValueError("Out-of-order answer")
                            answer = validate_answer(payload, config.profile)
                            observation.validated_answer_ms = now_ms
                            observation.abstained = answer.abstained
                            answer_seen = True
                        elif event == "done":
                            if not answer_seen or payload != {}:
                                raise ValueError("Done without validated answer")
                            done = True
                        else:
                            raise ValueError("Unknown SSE event")
                        event, data = "", []
                    if not done or data:
                        raise ValueError("Truncated SSE stream")
                observation.success = True
    except (TimeoutError, httpx.TimeoutException):
        observation.error = "timeout"
    except httpx.HTTPError:
        observation.error = "transport_error"
    except (ValueError, ValidationError):
        observation.error = "invalid_response"
    finally:
        observation.elapsed_ms = (time.perf_counter() - started) * 1000
    return observation


async def benchmark(
    config: LoadConfig, *, transport: httpx.AsyncBaseTransport | None = None
) -> dict[str, Any]:
    """Fixed worker pool; no retries, no credentials/body/URL in returned output."""
    config.validate()
    token = os.environ.get(config.token_env, "")
    if not token or token != token.strip() or "\r" in token or "\n" in token:
        raise ValueError("A nonempty bearer token is required through the configured environment")
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(
        headers=headers,
        timeout=config.timeout,
        limits=httpx.Limits(
            max_connections=config.concurrency, max_keepalive_connections=config.concurrency
        ),
        transport=transport,
        follow_redirects=False,
        trust_env=False,
    ) as client:
        warmup_errors: Counter[str] = Counter()
        for _ in range(config.warmup):
            result = await query_once(client, config)
            if not result.success:
                warmup_errors[result.error or "unknown"] += 1
        observations: list[Observation] = []
        claimed = 0
        start = time.perf_counter()
        deadline = start + config.duration if config.duration is not None else None

        async def worker() -> None:
            nonlocal claimed
            while True:
                # No await between checking and claiming: atomic in this event loop.
                if deadline is not None and time.perf_counter() >= deadline:
                    return
                if config.requests is not None and claimed >= config.requests:
                    return
                claimed += 1
                observations.append(await query_once(client, config))

        await asyncio.gather(*(worker() for _ in range(config.concurrency)))
        elapsed_seconds = time.perf_counter() - start
    successes = [observation for observation in observations if observation.success]
    errors = Counter(
        observation.error or "unknown" for observation in observations if not observation.success
    )
    return {
        "schema_version": "1",
        "workload": {
            "profile": config.profile.value,
            "stream": config.stream,
            "top_k": config.top_k,
            "concurrency": config.concurrency,
            "request_budget": config.requests,
            "duration_budget_seconds": config.duration,
            "timeout_seconds": config.timeout,
            "warmup_requests": config.warmup,
            "arrival_model": "closed-loop; fixed concurrency; no retries",
        },
        "attempted": len(observations),
        "succeeded": len(successes),
        "success_rate": len(successes) / len(observations) if observations else None,
        "elapsed_seconds": elapsed_seconds,
        "throughput_requests_per_second": len(observations) / elapsed_seconds,
        "goodput_answers_per_second": len(successes) / elapsed_seconds,
        "errors": dict(sorted(errors.items())),
        "warmup_errors": dict(sorted(warmup_errors.items())),
        "abstained_successes": sum(observation.abstained is True for observation in successes),
        "latency_ms": {
            "all_request_completion": latency_summary(
                [observation.elapsed_ms for observation in observations]
            ),
            "successful_request_completion": latency_summary(
                [observation.elapsed_ms for observation in successes]
            ),
            "ttfe_successful_sse": latency_summary(
                [
                    observation.ttfe_ms
                    for observation in successes
                    if observation.ttfe_ms is not None
                ]
            ),
            "validated_answer_successful": latency_summary(
                [
                    observation.validated_answer_ms
                    for observation in successes
                    if observation.validated_answer_ms is not None
                ]
            ),
        },
        "latency_sample_counts": {
            "all_request_completion": len(observations),
            "successful_request_completion": len(successes),
            "ttfe_successful_sse": sum(o.ttfe_ms is not None for o in successes),
            "validated_answer_successful": sum(
                o.validated_answer_ms is not None for o in successes
            ),
        },
        "warning": (
            "TTFE is a complete evidence event, not a token or final answer. "
            "Validated-answer latency checks wire schema, not semantic truth. "
            "Success latency excludes failures; all-request completion and errors are separate. "
            "Duration stops new arrivals; in-flight requests drain within timeout. "
            "Closed-loop results can hide coordinated omission; not an open-loop capacity claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded authenticated HTTP load benchmark")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--question", default=LoadConfig.question)
    parser.add_argument("--profile", type=Profile, choices=list(Profile), default=Profile.CHAT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    budgets = parser.add_mutually_exclusive_group()
    budgets.add_argument("--requests", type=int)
    budgets.add_argument("--duration", type=float)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--token-env", default="LEDGER_BENCHMARK_TOKEN")
    parser.add_argument("--min-success-rate", type=float, default=0.99)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.min_success_rate) or not 0 <= args.min_success_rate <= 1:
        parser.exit(2, "Minimum success rate must be finite in [0, 1].\n")
    config = LoadConfig(
        url=args.url,
        question=args.question,
        profile=args.profile,
        top_k=args.top_k,
        concurrency=args.concurrency,
        requests=args.requests
        if args.requests is not None
        else (None if args.duration is not None else 100),
        duration=args.duration,
        warmup=args.warmup,
        timeout=args.timeout,
        stream=args.stream,
        token_env=args.token_env,
    )
    try:
        result = asyncio.run(benchmark(config))
        serialized = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n")
        print(serialized)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Benchmark configuration/output failure ({type(exc).__name__}).\n")
    if result["success_rate"] is None or result["success_rate"] < args.min_success_rate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
