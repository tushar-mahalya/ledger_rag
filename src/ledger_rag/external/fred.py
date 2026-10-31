"""FRED/ALFRED observations as they were known at an as-of date.

One observations request asks for every real-time period (``realtime_start=1776-07-04``
to ``9999-12-31``): FRED clips real-time fields to the requested window, so only the full
history reveals the vintage date on which a value first became current and whether a
later vintage revised it. The value known at the cutoff is the row whose real-time
period contains the cutoff; ``"."`` is FRED's missing-value sentinel.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from ledger_rag.core.provenance import plausible
from ledger_rag.external.errors import ExternalDataError
from ledger_rag.external.http import ExternalHttp, parse_json

FRED_API = "https://api.stlouisfed.org/fred"
FRED_SITE = "https://fred.stlouisfed.org"
SERIES_PATTERN = re.compile(r"^[A-Z0-9_]{1,40}$")
API_KEY_PATTERN = re.compile(r"^[a-z0-9]{32}$")
REALTIME_START = "1776-07-04"
REALTIME_END = "9999-12-31"
OBSERVATION_ROW_LIMIT = 20_000
MAX_RANGE_DAYS = 3660
# Furthest an explicit date may fall after the observation that covers it.
GAP_DAYS: Mapping[str, int] = {"D": 7, "W": 8, "BW": 15, "M": 31, "Q": 92, "SA": 183, "A": 366}
STALE_DAYS: Mapping[str, int] = {
    "D": 14,
    "W": 30,
    "BW": 45,
    "M": 95,
    "Q": 200,
    "SA": 380,
    "A": 500,
}
# FRED dates monthly-and-longer observations by the first day of the period.
PERIOD_MONTHS: Mapping[str, int] = {"M": 1, "Q": 3, "SA": 6, "A": 12}


@dataclass(frozen=True)
class SeriesMeta:
    id: str
    title: str
    frequency: str
    frequency_short: str
    units: str
    units_short: str
    seasonal_adjustment_short: str

    @property
    def gap_days(self) -> int:
        return GAP_DAYS.get(self.frequency_short, 31)

    def period_bounds(self, day: date) -> tuple[date | None, date]:
        months = PERIOD_MONTHS.get(self.frequency_short)
        if months is None:
            return None, day
        year, month = divmod(day.month - 1 + months, 12)
        return day, date(day.year + year, month + 1, 1) - timedelta(days=1)


@dataclass(frozen=True)
class Vintage:
    day: date
    value: Decimal | None
    realtime_start: date
    realtime_end: date


@dataclass(frozen=True)
class MacroObservation:
    series: SeriesMeta
    day: date
    value: Decimal
    vintage: date
    revised_before: bool
    revised_after: date | None
    stale: bool


@dataclass(frozen=True)
class MacroResult:
    series: SeriesMeta
    observations: list[MacroObservation]
    missing: list[str]


def normalize_series(text: str) -> str:
    series = text.strip().upper()
    if not SERIES_PATTERN.fullmatch(series):
        raise ExternalDataError(
            "invalid_request", "series_id must be a FRED series ID such as FEDFUNDS or GDP"
        )
    return series


def _date(value: Any) -> date:
    if not isinstance(value, str):
        raise ValueError("date is not a string")
    if value == REALTIME_END:
        return date.max
    return date.fromisoformat(value)


def _value(value: Any) -> Decimal | None:
    if value == ".":
        return None
    if not isinstance(value, str):
        raise ValueError("value is not a string")
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise ValueError("value is not a number") from None
    if not plausible(number):
        raise ValueError("value is out of range")
    return number


def parse_series(payload: Any) -> SeriesMeta:
    rows = payload.get("seriess") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        raise ValueError("series payload has no series")
    row = rows[0]

    def text(key: str, limit: int) -> str:
        value = row.get(key)
        return value.strip()[:limit] if isinstance(value, str) else ""

    series_id = text("id", 40)
    if not SERIES_PATTERN.fullmatch(series_id):
        raise ValueError("series payload has an invalid id")
    return SeriesMeta(
        id=series_id,
        title=text("title", 300) or series_id,
        frequency=text("frequency", 60) or "Unknown frequency",
        frequency_short=text("frequency_short", 8),
        units=text("units", 120) or "Units",
        units_short=text("units_short", 60),
        seasonal_adjustment_short=text("seasonal_adjustment_short", 12),
    )


def parse_observations(payload: Any) -> list[Vintage]:
    rows = payload.get("observations") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("observations payload has no observations")
    count = payload.get("count")
    if isinstance(count, int) and count > len(rows):
        raise ExternalDataError(
            "too_large", "That range has too many revisions; request a shorter range or dates"
        )
    vintages = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("observation is not an object")
        vintages.append(
            Vintage(
                day=_date(row.get("date")),
                value=_value(row.get("value")),
                realtime_start=_date(row.get("realtime_start")),
                realtime_end=_date(row.get("realtime_end")),
            )
        )
    return vintages


def fred_error(status: int, raw: bytes) -> ExternalDataError | None:
    """Map FRED's JSON errors (``error_code``/``error_message``) to safe kinds."""
    if status == 429:
        return ExternalDataError("rate_limited", "FRED is rate limiting requests")
    if status != 400:
        return None
    try:
        payload = parse_json(raw)
    except ValueError:
        return None
    message = payload.get("error_message", "") if isinstance(payload, dict) else ""
    message = message.lower() if isinstance(message, str) else ""
    if "api_key" in message or "api key" in message:
        return ExternalDataError("forbidden", "FRED rejected the configured API key")
    if "does not exist in alfred" in message:
        return ExternalDataError(
            "invalid_request",
            "That series has no ALFRED vintage history, so values known at the as-of date "
            "cannot be established",
        )
    if "does not exist" in message:
        return ExternalDataError("not_found", "FRED has no series with that ID")
    return ExternalDataError("invalid_request", "FRED rejected the request parameters")


class FredClient:
    """Read-only FRED/ALFRED access; the key is sent only to api.stlouisfed.org."""

    def __init__(self, http: ExternalHttp, api_key: str) -> None:
        self.http = http
        self._api_key = api_key

    def _params(self, **params: str) -> dict[str, str]:
        return {**params, "api_key": self._api_key, "file_type": "json"}

    async def series(self, series_id: str, as_of: date) -> SeriesMeta:
        day = as_of.isoformat()
        return await self.http.get(
            f"{FRED_API}/series",
            key=f"fred:series:{series_id}:{day}",
            endpoint="series",
            params=self._params(series_id=series_id, realtime_start=day, realtime_end=day),
            parse=lambda raw: parse_series(parse_json(raw)),
            on_error=fred_error,
        )

    async def vintages(self, series_id: str, start: date, end: date) -> list[Vintage]:
        return await self.http.get(
            f"{FRED_API}/series/observations",
            key=f"fred:observations:{series_id}:{start}:{end}",
            endpoint="observations",
            params=self._params(
                series_id=series_id,
                realtime_start=REALTIME_START,
                realtime_end=REALTIME_END,
                observation_start=start.isoformat(),
                observation_end=end.isoformat(),
                sort_order="asc",
                limit=str(OBSERVATION_ROW_LIMIT),
            ),
            parse=lambda raw: parse_observations(parse_json(raw)),
            on_error=fred_error,
        )

    async def lookup(
        self,
        series_id: str,
        *,
        as_of: date,
        start: date | None = None,
        end: date | None = None,
        dates: list[date] | None = None,
        max_observations: int,
    ) -> MacroResult:
        series_id = normalize_series(series_id)
        meta = await self.series(series_id, as_of)
        gap = timedelta(days=meta.gap_days)
        if dates:
            window = (min(dates) - gap, min(max(dates), as_of))
        elif start is not None:
            window = (start, min(end or as_of, as_of))
        else:
            window = (as_of - gap * 4, as_of)
        if window[0] > window[1]:
            raise ExternalDataError(
                "invalid_request", f"No observations can be known on or before {as_of}"
            )
        if (window[1] - window[0]).days > MAX_RANGE_DAYS:
            raise ExternalDataError("invalid_request", "Date ranges are limited to ten years")
        known = known_values(await self.vintages(series_id, *window), as_of)
        missing: list[str] = []
        chosen: list[date] = []
        if dates:
            for wanted in dates:
                covering = [
                    day for day, row in known.items() if wanted - gap < day <= wanted and row
                ]
                if covering:
                    chosen.append(max(covering))
                else:
                    missing.append(wanted.isoformat())
        elif start is not None:
            chosen = [day for day, row in known.items() if row and window[0] <= day <= window[1]]
            missing = [
                day.isoformat()
                for day, row in known.items()
                if not row and window[0] <= day <= window[1]
            ]
            if len(chosen) > max_observations:
                raise ExternalDataError(
                    "invalid_request",
                    f"That range has {len(chosen)} observations; at most {max_observations} "
                    "per call, so narrow the range or pass specific dates",
                )
        else:
            available = [day for day, row in known.items() if row]
            if available:
                chosen = [max(available)]
            else:
                missing = ["latest"]
        observations = []
        stale_after = STALE_DAYS.get(meta.frequency_short, 95)
        for day in dict.fromkeys(chosen):
            row = known[day]
            assert row is not None
            observations.append(
                MacroObservation(
                    series=meta,
                    day=day,
                    value=row.value,
                    vintage=row.vintage,
                    revised_before=row.revised_before,
                    revised_after=row.revised_after,
                    stale=not dates and start is None and (as_of - day).days > stale_after,
                )
            )
        return MacroResult(series=meta, observations=observations, missing=missing)


@dataclass(frozen=True)
class KnownValue:
    value: Decimal
    vintage: date
    revised_before: bool
    revised_after: date | None


def known_values(vintages: list[Vintage], as_of: date) -> dict[date, KnownValue | None]:
    """Per observation date: the value current at ``as_of`` (None when FRED had ".").

    Dates first published after ``as_of`` are omitted entirely.
    """
    by_day: dict[date, list[Vintage]] = {}
    for row in vintages:
        by_day.setdefault(row.day, []).append(row)
    known: dict[date, KnownValue | None] = {}
    for day in sorted(by_day):
        rows = sorted(by_day[day], key=lambda row: row.realtime_start)
        current = next(
            (row for row in rows if row.realtime_start <= as_of <= row.realtime_end), None
        )
        if current is None:
            continue
        if current.value is None:
            known[day] = None
            continue
        earlier = [row for row in rows if row.realtime_end < current.realtime_start]
        later = next(
            (row for row in rows if row.realtime_start > as_of and row.value != current.value),
            None,
        )
        known[day] = KnownValue(
            value=current.value,
            vintage=current.realtime_start,
            revised_before=any(row.value != current.value for row in earlier),
            revised_after=later.realtime_start if later else None,
        )
    return known


def series_url(series_id: str) -> str:
    return f"{FRED_SITE}/series/{series_id}"
