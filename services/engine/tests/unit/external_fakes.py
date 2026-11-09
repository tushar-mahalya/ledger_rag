"""Offline SEC and FRED upstreams served from ``tests/fixtures/external``."""

import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr

from ledger_rag.core.config import Settings
from ledger_rag.external.service import ExternalData

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "external"
FRED_KEY = "0123456789abcdef0123456789abcdef"  # gitleaks:allow (fake test key)
NOW = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
Handler = Callable[[httpx.Request], httpx.Response]


def fixture(provider: str, name: str) -> Any:
    return json.loads((FIXTURES / provider / name).read_text())


class FakeUpstream:
    """Routes data.sec.gov, www.sec.gov and api.stlouisfed.org to fixture files."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.overrides: dict[str, Handler] = {}

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def count(self, fragment: str) -> int:
        return sum(fragment in request.url.path for request in self.requests)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        for fragment, handler in self.overrides.items():
            if fragment in path:
                return handler(request)
        if request.url.host in {"data.sec.gov", "www.sec.gov"}:
            return self._sec(path)
        if request.url.host == "api.stlouisfed.org":
            return self._fred(request)
        return httpx.Response(599)

    def _sec(self, path: str) -> httpx.Response:
        name: str | None = None
        if path == "/files/company_tickers.json":
            name = "company_tickers.json"
        elif path.startswith("/api/xbrl/companyconcept/"):
            cik, taxonomy, concept = path.removeprefix("/api/xbrl/companyconcept/").split("/")
            name = f"concept_{cik[3:]}_{taxonomy}_{concept}"
        elif path.startswith("/api/xbrl/companyfacts/"):
            name = f"companyfacts_{path.rsplit('/', 1)[1][3:]}"
        elif path.startswith("/submissions/"):
            name = f"submissions_{path.rsplit('/', 1)[1][3:]}"
        if name and (FIXTURES / "sec" / name).is_file():
            return httpx.Response(
                200,
                content=(FIXTURES / "sec" / name).read_bytes(),
                headers={"content-type": "application/json"},
            )
        return httpx.Response(
            404,
            content=(FIXTURES / "sec" / "not_found.xml").read_bytes(),
            headers={"content-type": "application/xml"},
        )

    def _fred(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if params.get("api_key") != FRED_KEY:
            return httpx.Response(400, json=fixture("fred", "error_api_key.json"))
        series = params.get("series_id", "")
        if not (FIXTURES / "fred" / f"series_{series}.json").is_file():
            return httpx.Response(400, json=fixture("fred", "error_no_series.json"))
        if request.url.path == "/fred/series":
            return httpx.Response(200, json=fixture("fred", f"series_{series}.json"))
        if not (FIXTURES / "fred" / f"observations_{series}.json").is_file():
            return httpx.Response(400, json=fixture("fred", "error_not_in_alfred.json"))
        payload = fixture("fred", f"observations_{series}.json")
        start = params.get("observation_start", "0000-00-00")
        end = params.get("observation_end", "9999-99-99")
        rows = [row for row in payload["observations"] if start <= row["date"] <= end]
        payload.update(observations=rows, count=len(rows))
        return httpx.Response(200, json=payload)


def external_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "demo_token": SecretStr("test-only-token"),
        "external_data_enabled": True,
        "fred_api_key": SecretStr(FRED_KEY),
    }
    values.update(overrides)
    return Settings(**values)


def external_data(
    upstream: FakeUpstream | None = None, *, now: datetime = NOW, **overrides: Any
) -> ExternalData:
    return ExternalData(
        external_settings(**overrides),
        transport=(upstream or FakeUpstream()).transport,
        now=lambda: now,
    )


def day(text: str) -> date:
    return date.fromisoformat(text)
