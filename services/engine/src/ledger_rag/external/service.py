"""Facade over the SEC and FRED clients: configuration, authorization, budgets, breakers.

``ExternalData.from_settings`` returns ``None`` unless ``LEDGER_EXTERNAL_DATA_ENABLED``
is set; each provider can still be disabled on its own and reports why in ``status()``.
"""

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Literal, TypeVar

import httpx
import structlog
from pydantic import BaseModel, ConfigDict

from ledger_rag.core.config import Settings
from ledger_rag.core.models import Principal
from ledger_rag.core.retry import RetryPolicy
from ledger_rag.external.breaker import BreakerState, CircuitBreaker
from ledger_rag.external.errors import ExternalDataError
from ledger_rag.external.fred import (
    API_KEY_PATTERN,
    FredClient,
    MacroObservation,
    SeriesMeta,
    Vintage,
    normalize_series,
)
from ledger_rag.external.http import ExternalHttp, ResponseCache, SlidingWindowLimiter
from ledger_rag.external.normalize import ExternalFact, fred_fact, sec_fact
from ledger_rag.external.sec import (
    ACCESSION_PATTERN,
    Company,
    ConceptFacts,
    PeriodSpec,
    SecClient,
    Selection,
    SelectionMiss,
    parse_period,
    select_fact,
)

logger = structlog.get_logger(__name__)
T = TypeVar("T")
Provider = Literal["sec", "fred"]
PROVIDER_NAMES: dict[Provider, str] = {"sec": "SEC EDGAR", "fred": "FRED"}
_SEC_EVIDENCE = re.compile(
    r"^sec:([0-9]{10}):(us-gaap|ifrs-full|dei|srt):([A-Za-z][A-Za-z0-9_]{0,131}):"
    r"([0-9]{10}-[0-9]{2}-[0-9]{6}):([0-9]{4}-[0-9]{2}-[0-9]{2})"
    r"(?:_([0-9]{4}-[0-9]{2}-[0-9]{2}))?(?::([A-Za-z0-9._:/-]{1,60}))?$"
)
_FRED_EVIDENCE = re.compile(
    r"^fred:([A-Z0-9_]{1,40}):([0-9]{4}-[0-9]{2}-[0-9]{2}):([0-9]{4}-[0-9]{2}-[0-9]{2})$"
)


class ProviderStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool
    reason: str | None = None
    circuit: BreakerState | None = None


class ExternalStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool
    sec: ProviderStatus
    fred: ProviderStatus

    @classmethod
    def disabled(cls) -> "ExternalStatus":
        off = ProviderStatus(enabled=False, reason="external data is disabled")
        return cls(enabled=False, sec=off, fred=off)


@dataclass(frozen=True)
class Miss:
    item: str
    period: str
    reason: str

    def payload(self) -> dict[str, str]:
        return {"item": self.item, "period": self.period, "reason": self.reason}


@dataclass(frozen=True)
class Lookup:
    """Facts plus per-item misses; misses are normal results, not errors."""

    facts: list[ExternalFact]
    missing: list[Miss]
    subject: str


def fred_key(settings: Settings) -> tuple[str | None, str | None]:
    """(key, reason it cannot be used); the key never appears in the reason."""
    if settings.fred_api_key is None:
        return None, (
            "no FRED API key is configured (LEDGER_FRED_API_KEY; ledgerctl set LEDGER_FRED_API_KEY)"
        )
    key = settings.fred_api_key.get_secret_value().strip()
    if not API_KEY_PATTERN.fullmatch(key):
        return None, "the FRED API key is malformed (expected 32 lowercase letters or digits)"
    return key, None


class ExternalData:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.settings = settings
        self._now = now
        self._timeout = settings.external_timeout_seconds
        self._cache = ResponseCache(
            settings.external_cache_ttl_seconds, settings.external_cache_max_bytes
        )
        policy = RetryPolicy.from_settings(settings)
        self._clients: list[httpx.AsyncClient] = []
        self._reasons: dict[Provider, str | None] = {"sec": None, "fred": None}
        self._breakers = {
            provider: CircuitBreaker(
                PROVIDER_NAMES[provider],
                settings.external_breaker_failures,
                settings.external_breaker_reset_seconds,
            )
            for provider in PROVIDER_NAMES
        }

        def client(user_agent: str) -> httpx.AsyncClient:
            created = httpx.AsyncClient(
                transport=transport,
                timeout=httpx.Timeout(self._timeout),
                follow_redirects=False,
                headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            )
            self._clients.append(created)
            return created

        self.sec: SecClient | None = None
        if not settings.sec_enabled:
            self._reasons["sec"] = "disabled by LEDGER_SEC_ENABLED"
        else:
            self.sec = SecClient(
                ExternalHttp(
                    provider="SEC EDGAR",
                    client=client(settings.sec_user_agent),
                    limiter=SlidingWindowLimiter(settings.sec_max_requests_per_second, 1.0),
                    cache=self._cache,
                    policy=policy,
                    max_bytes=settings.external_max_response_bytes,
                    timeout_seconds=self._timeout,
                )
            )
        self.fred: FredClient | None = None
        key, reason = fred_key(settings)
        if not settings.fred_enabled:
            self._reasons["fred"] = "disabled by LEDGER_FRED_ENABLED"
        elif key is None:
            self._reasons["fred"] = reason
        else:
            self.fred = FredClient(
                ExternalHttp(
                    provider="FRED",
                    client=client("Ledger RAG"),
                    limiter=SlidingWindowLimiter(settings.fred_max_requests_per_minute, 60.0),
                    cache=self._cache,
                    policy=policy,
                    max_bytes=settings.external_max_response_bytes,
                    timeout_seconds=self._timeout,
                ),
                key,
            )

    @classmethod
    def from_settings(cls, settings: Settings) -> "ExternalData | None":
        return cls(settings) if settings.external_data_enabled else None

    def status(self) -> ExternalStatus:
        def provider(name: Provider, enabled: bool) -> ProviderStatus:
            return ProviderStatus(
                enabled=enabled and self._reasons[name] is None,
                reason=self._reasons[name],
                circuit=self._breakers[name].state if enabled else None,
            )

        return ExternalStatus(
            enabled=True,
            sec=provider("sec", self.sec is not None),
            fred=provider("fred", self.fred is not None),
        )

    def providers(self, principal: Principal | None) -> list[Provider]:
        """Providers this principal may use right now (empty when not entitled)."""
        if principal is None or not self.entitled(principal):
            return []
        available: list[Provider] = []
        if self.sec is not None and self._reasons["sec"] is None:
            available.append("sec")
        if self.fred is not None and self._reasons["fred"] is None:
            available.append("fred")
        return available

    def entitled(self, principal: Principal) -> bool:
        settings = self.settings
        if "reader" not in principal.roles:
            return False
        if settings.external_required_group and (
            settings.external_required_group not in principal.groups
        ):
            return False
        return not settings.external_allowed_tenants or (
            principal.tenant_id in settings.external_allowed_tenants
        )

    def as_of(self, requested: date | None) -> date:
        today = self._now().date()
        return min(requested, today) if requested is not None else today

    async def _guard(self, provider: Provider, call: Callable[[], Awaitable[T]]) -> T:
        breaker = self._breakers[provider]
        breaker.before()
        try:
            async with asyncio.timeout(self._timeout):
                result = await call()
        except TimeoutError:
            error = ExternalDataError(
                "timeout", f"{PROVIDER_NAMES[provider]} did not respond in time"
            )
            breaker.failure(error)
            raise error from None
        except ExternalDataError as error:
            if provider == "fred" and error.kind == "forbidden":
                self._reasons["fred"] = "FRED rejected the configured API key"
            breaker.failure(error)
            raise
        except asyncio.CancelledError:
            breaker.release()
            raise
        except Exception as exc:
            logger.error(
                "external_lookup_failed", provider=provider, error_class=type(exc).__name__
            )
            unexpected = ExternalDataError(
                "upstream", f"{PROVIDER_NAMES[provider]} lookup failed unexpectedly"
            )
            breaker.failure(unexpected)
            raise unexpected from None
        breaker.success()
        return result

    def _require(self, provider: Provider) -> None:
        if (self.sec if provider == "sec" else self.fred) is None or self._reasons[provider]:
            raise ExternalDataError(
                "disabled", f"{PROVIDER_NAMES[provider]} lookups are not available"
            )

    async def company_facts(
        self,
        company: str,
        concepts: list[str],
        periods: list[str],
        *,
        as_of: date,
        unit: str | None = None,
    ) -> Lookup:
        self._require("sec")
        assert self.sec is not None
        sec = self.sec
        specs = [parse_period(period) for period in periods]
        if len(concepts) * len(specs) > self.settings.external_max_facts_per_call:
            raise ExternalDataError(
                "invalid_request",
                f"At most {self.settings.external_max_facts_per_call} concept-period pairs "
                "per call",
            )

        async def run() -> Lookup:
            entity = await sec.resolve(company)
            facts: list[ExternalFact] = []
            missing: list[Miss] = []
            retrieved = self._now()
            for concept in dict.fromkeys(concepts):
                try:
                    found = await sec.find_concept(entity.cik, concept)
                except ExternalDataError as error:
                    if error.kind not in {"not_found", "invalid_request"}:
                        raise
                    missing += [Miss(concept, spec.label, error.message) for spec in specs]
                    continue
                for spec in dict.fromkeys(specs):
                    try:
                        selection = select_fact(found, spec, as_of, unit)
                    except SelectionMiss as miss:
                        missing.append(Miss(concept, spec.label, str(miss)))
                        continue
                    facts.append(sec_fact(selection, as_of=as_of, retrieved_at=retrieved))
            return Lookup(facts=facts, missing=missing, subject=_subject(entity))

        return await self._guard("sec", run)

    async def macro_series(
        self,
        series_id: str,
        *,
        as_of: date,
        start: date | None = None,
        end: date | None = None,
        dates: list[date] | None = None,
    ) -> Lookup:
        self._require("fred")
        assert self.fred is not None
        fred = self.fred
        series = normalize_series(series_id)

        async def run() -> Lookup:
            result = await fred.lookup(
                series,
                as_of=as_of,
                start=start,
                end=end,
                dates=dates,
                max_observations=self.settings.external_max_facts_per_call,
            )
            retrieved = self._now()
            meta = result.series
            return Lookup(
                facts=[
                    fred_fact(observation, as_of=as_of, retrieved_at=retrieved)
                    for observation in result.observations
                ],
                missing=[
                    Miss(series, day, "no value known at the as-of date") for day in result.missing
                ],
                subject=f"{meta.id}: {meta.title} ({meta.units}, {meta.frequency})",
            )

        return await self._guard("fred", run)

    async def read(self, evidence_id: str, *, as_of: date) -> ExternalFact:
        """Re-fetch one evidence ID; anything unknown or filed after ``as_of`` is not found."""
        sec_match = _SEC_EVIDENCE.fullmatch(evidence_id)
        if sec_match is not None:
            self._require("sec")
            return await self._guard("sec", lambda: self._read_sec(sec_match, as_of))
        fred_match = _FRED_EVIDENCE.fullmatch(evidence_id)
        if fred_match is not None:
            self._require("fred")
            return await self._guard("fred", lambda: self._read_fred(fred_match, as_of))
        raise ExternalDataError("not_found", "Evidence not found")

    async def _read_sec(self, match: re.Match[str], as_of: date) -> ExternalFact:
        assert self.sec is not None
        cik, taxonomy, concept, accession = (match.group(index) for index in range(1, 5))
        first, second = match.group(5), match.group(6)
        start = date.fromisoformat(first) if second else None
        end = date.fromisoformat(second or first)
        if not ACCESSION_PATTERN.fullmatch(accession):
            raise ExternalDataError("not_found", "Evidence not found")
        try:
            found = await self.sec.concept(int(cik), taxonomy, concept)
        except ExternalDataError as error:
            if error.kind == "not_found":
                raise ExternalDataError("not_found", "Evidence not found") from None
            raise
        selection = _pinned_selection(found, accession, (start, end), as_of, match.group(7))
        return sec_fact(selection, as_of=as_of, retrieved_at=self._now())

    async def _read_fred(self, match: re.Match[str], as_of: date) -> ExternalFact:
        assert self.fred is not None
        series, day, vintage = match.group(1), match.group(2), match.group(3)
        observed, published = date.fromisoformat(day), date.fromisoformat(vintage)
        if published > as_of:
            raise ExternalDataError("not_found", "Evidence not found")
        meta = await self.fred.series(series, as_of)
        rows = await self.fred.vintages(series, observed, observed)
        return fred_fact(
            _pinned_observation(meta, rows, observed, published),
            as_of=as_of,
            retrieved_at=self._now(),
        )

    async def aclose(self) -> None:
        try:
            await asyncio.gather(
                *(
                    provider.http.aclose()
                    for provider in (self.sec, self.fred)
                    if provider is not None
                )
            )
        finally:
            await asyncio.gather(*(client.aclose() for client in self._clients))


def _subject(company: Company) -> str:
    tickers = "/".join(company.tickers[:4])
    return f"{company.name} (CIK {company.padded}{', ' + tickers if tickers else ''})"


def _pinned_selection(
    found: ConceptFacts,
    accession: str,
    period: tuple[date | None, date],
    as_of: date,
    unit: str | None = None,
) -> Selection:
    facts = [
        fact
        for fact in found.facts
        if fact.accn == accession
        and fact.period == period
        and fact.filed <= as_of
        and (unit is None or fact.unit == unit)
    ]
    if len({(fact.unit, fact.value) for fact in facts}) != 1:
        raise ExternalDataError("not_found", "Evidence not found")
    fact = facts[0]
    same = [other for other in found.facts if other.period == period and other.unit == fact.unit]
    later = sorted(
        (other for other in same if (other.filed, other.accn) > (fact.filed, fact.accn)),
        key=lambda other: (other.filed, other.accn),
    )
    start, end = period
    label = end.isoformat() if start is None else f"{start.isoformat()}/{end.isoformat()}"
    return Selection(
        fact=fact,
        concept=found,
        spec=PeriodSpec("date", day=end) if start is None else PeriodSpec("latest"),
        period_label=label,
        fiscal_year=None,
        fiscal_period=None,
        restated=any(
            other.value != fact.value
            for other in same
            if (other.filed, other.accn) < (fact.filed, fact.accn)
        ),
        superseded_by=next((other for other in later if other.value != fact.value), None),
        stale=False,
    )


def _pinned_observation(
    meta: SeriesMeta, rows: list[Vintage], day: date, vintage: date
) -> MacroObservation:
    ordered = sorted((row for row in rows if row.day == day), key=lambda row: row.realtime_start)
    pinned = next((row for row in ordered if row.realtime_start == vintage), None)
    if pinned is None or pinned.value is None:
        raise ExternalDataError("not_found", "Evidence not found")
    later = next(
        (
            row
            for row in ordered
            if row.realtime_start > pinned.realtime_start and row.value != pinned.value
        ),
        None,
    )
    return MacroObservation(
        series=meta,
        day=day,
        value=pinned.value,
        vintage=pinned.realtime_start,
        revised_before=any(
            row.value != pinned.value
            for row in ordered
            if row.realtime_start < pinned.realtime_start
        ),
        revised_after=later.realtime_start if later else None,
        stale=False,
    )


@dataclass
class ExternalSession:
    """One chat turn's view: fixed principal, as-of cutoff and lookup budget."""

    data: ExternalData
    principal: Principal
    as_of: date
    providers: list[Provider]
    calls: int = field(default=0)

    def spend(self) -> None:
        if self.calls >= self.data.settings.external_max_calls_per_turn:
            raise ExternalDataError(
                "budget_exhausted",
                "The external lookup budget for this answer is used up; answer with the "
                "evidence already gathered",
            )
        self.calls += 1
