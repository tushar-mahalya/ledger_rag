import asyncio
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from ledger_rag.ingestion.bulk import IngestSpec
from ledger_rag.ingestion.tei import embedding_server


@pytest.fixture
def process(monkeypatch: pytest.MonkeyPatch) -> Mock:
    child = Mock(returncode=None, wait=AsyncMock(return_value=0), stdout=None, stderr=None)
    start = AsyncMock(return_value=child)
    monkeypatch.setattr("ledger_rag.ingestion.tei.asyncio.create_subprocess_exec", start)
    child.start = start
    return child


def client(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "ledger_rag.ingestion.tei.httpx.AsyncClient",
        lambda **kwargs: real_client(
            **kwargs, transport=httpx.MockTransport(lambda request: httpx.Response(status))
        ),
    )


async def test_router_is_private_pinned_and_key_is_not_in_argv(
    monkeypatch: pytest.MonkeyPatch, process: Mock
) -> None:
    client(monkeypatch, 200)
    monkeypatch.setenv("LEDGER_QDRANT_API_KEY", "database-secret")
    spec = IngestSpec(collection="new", model_revision="a" * 40)
    async with embedding_server(spec, "per-job-key"):
        argv = process.start.call_args.args
        environment = process.start.call_args.kwargs["env"]
        assert argv[argv.index("--hostname") + 1] == "127.0.0.1"
        assert argv[argv.index("--revision") + 1] == "a" * 40
        assert argv[argv.index("--dtype") + 1] == "float16"
        assert "per-job-key" not in argv
        assert environment["API_KEY"] == "per-job-key"
        assert environment["AUTO_TRUNCATE"] == "false"
        assert "LEDGER_QDRANT_API_KEY" not in environment
        assert process.start.call_args.kwargs["stdout"] == asyncio.subprocess.PIPE
        assert process.start.call_args.kwargs["stderr"] == asyncio.subprocess.PIPE
    process.terminate.assert_called_once()
    process.wait.assert_awaited_once()


async def test_startup_error_is_sanitized_and_process_is_stopped(
    monkeypatch: pytest.MonkeyPatch, process: Mock
) -> None:
    client(monkeypatch, 401)
    with pytest.raises(RuntimeError, match="HTTP 401"):
        async with embedding_server(IngestSpec(collection="new", model_revision="a" * 40), "key"):
            pytest.fail("unhealthy server must not yield")
    process.terminate.assert_called_once()


async def test_early_exit_is_explicit(monkeypatch: pytest.MonkeyPatch, process: Mock) -> None:
    client(monkeypatch, 200)
    process.returncode = 7
    with pytest.raises(RuntimeError, match="code 7"):
        async with embedding_server(IngestSpec(collection="new", model_revision="a" * 40), "key"):
            pytest.fail("dead server must not yield")
    process.terminate.assert_not_called()


async def test_timeout_and_cancellation_cleanup(
    monkeypatch: pytest.MonkeyPatch, process: Mock
) -> None:
    client(monkeypatch, 503)
    with pytest.raises(TimeoutError, match="startup deadline"):
        async with embedding_server(
            IngestSpec(collection="new", model_revision="a" * 40), "key", startup_timeout=0.01
        ):
            pytest.fail("unready server must not yield")
    process.terminate.assert_called_once()


async def test_cancelled_ingestion_stops_router(
    monkeypatch: pytest.MonkeyPatch, process: Mock
) -> None:
    client(monkeypatch, 200)
    with pytest.raises(asyncio.CancelledError):
        async with embedding_server(IngestSpec(collection="new", model_revision="a" * 40), "key"):
            raise asyncio.CancelledError
    process.terminate.assert_called_once()


async def test_unresponsive_router_is_killed(
    monkeypatch: pytest.MonkeyPatch, process: Mock
) -> None:
    client(monkeypatch, 200)
    process.wait.side_effect = [TimeoutError, 0]
    async with embedding_server(IngestSpec(collection="new", model_revision="a" * 40), "key"):
        pass
    process.kill.assert_called_once()


async def test_router_output_is_forwarded_with_the_key_redacted(
    monkeypatch: pytest.MonkeyPatch, process: Mock, capsys: pytest.CaptureFixture[str]
) -> None:
    client(monkeypatch, 200)
    key = "per-job-key-0123456789"
    stdout, stderr = asyncio.StreamReader(), asyncio.StreamReader()
    process.stdout, process.stderr = stdout, stderr
    stdout.feed_data(f'Args {{ api_key: Some("{key}"), port: 8000 }}\n'.encode())
    stdout.feed_data(b"Ready\n")
    stderr.feed_data(f"warn {key}".encode())
    stdout.feed_eof()
    stderr.feed_eof()
    async with embedding_server(IngestSpec(collection="new", model_revision="a" * 40), key):
        pass
    captured = capsys.readouterr()
    assert key not in captured.out + captured.err
    assert 'api_key: Some("[REDACTED]")' in captured.out
    assert "Ready" in captured.out
    assert captured.err == "warn [REDACTED]"


async def test_overlong_router_line_never_leaks_the_key(
    monkeypatch: pytest.MonkeyPatch, process: Mock, capsys: pytest.CaptureFixture[str]
) -> None:
    client(monkeypatch, 200)
    monkeypatch.setattr("ledger_rag.ingestion.tei._LINE_LIMIT", 64)
    stdout = asyncio.StreamReader(limit=64)
    process.stdout = stdout
    stdout.feed_data(b"key " + b"x" * 100 + b"\n")
    stdout.feed_eof()
    async with embedding_server(IngestSpec(collection="new", model_revision="a" * 40), "x" * 100):
        pass
    assert "x" * 100 not in capsys.readouterr().out
