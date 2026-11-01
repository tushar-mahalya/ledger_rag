import os
from datetime import date

import pytest
from pydantic import SecretStr

from ledger_rag.core import retry
from ledger_rag.core.config import Settings
from ledger_rag.core.models import Document, Principal


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Network tests are opt-in so CI and local runs never depend on live upstreams."""
    if os.environ.get("LEDGER_LIVE_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="live upstream test; set LEDGER_LIVE_TESTS=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record upstream retry backoff instead of sleeping."""
    delays: list[float] = []

    async def record(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(retry, "_sleep", record)
    return delays


@pytest.fixture
def settings() -> Settings:
    return Settings(demo_token=SecretStr("test-only-token"))


@pytest.fixture
def principal() -> Principal:
    return Principal(subject="tester", tenant_id="demo", roles=["reader"], groups=[])


@pytest.fixture
def document() -> Document:
    return Document(
        id="doc-1",
        tenant_id="demo",
        title="Fictional fiscal disclosure",
        text="Revenue was USD 120 million in 2024. Operating margin was 15 percent.",
        source="urn:test:disclosure",
        published_at=date(2025, 2, 1),
    )
