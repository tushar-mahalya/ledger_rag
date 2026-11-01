from datetime import date

import pytest
from pydantic import SecretStr

from ledger_rag.core import retry
from ledger_rag.core.config import Settings
from ledger_rag.core.models import Document, Principal


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
