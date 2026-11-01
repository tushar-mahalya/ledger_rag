"""Typed provenance for evidence that does not come from the ingested corpus.

External facts (SEC XBRL filings, FRED/ALFRED observations) become ordinary citations
whose quote is a deterministic rendering of one exact number; this model carries the
structured facts behind that rendering so clients and auditors never parse prose.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

SourceKind = Literal["corpus", "sec_xbrl", "fred"]
ExternalSourceKind = Literal["sec_xbrl", "fred"]
# current: latest value known at the cutoff. stale: the period is old relative to its cadence.
# superseded: a filing or vintage published after the cutoff changed the value.
Freshness = Literal["current", "stale", "superseded"]

# Human-facing links only; API hosts (data.sec.gov, api.stlouisfed.org) are never linked.
SOURCE_HOSTS = frozenset({"www.sec.gov", "fred.stlouisfed.org"})
# The previous SEC identity ceiling plus a separator and its 60-character unit.
MAX_EXTERNAL_EVIDENCE_ID_CHARS = 261
EVIDENCE_ID_PATTERN = r"^(?:sec|fred):[A-Za-z0-9._:/-]{1,257}$"


MAX_MAGNITUDE = Decimal("1e30")
MAX_DECIMAL_PLACES = 20


def plausible(value: Decimal) -> bool:
    """Bounded like calculator values, so rendering a reported number stays short."""
    return (
        value.is_finite()
        and abs(value) < MAX_MAGNITUDE
        and int(value.as_tuple().exponent) >= -MAX_DECIMAL_PLACES
    )


def is_source_url(url: str) -> bool:
    parsed = urlsplit(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname in SOURCE_HOSTS
        and parsed.port is None
        and not parsed.username
        and not parsed.password
        and not parsed.fragment
    )


class Provenance(BaseModel):
    """Where an external value came from and what was known at the as-of cutoff."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_kind: ExternalSourceKind
    provider: Literal["SEC EDGAR", "FRED"]
    evidence_id: str = Field(
        min_length=5, max_length=MAX_EXTERNAL_EVIDENCE_ID_CHARS, pattern=EVIDENCE_ID_PATTERN
    )
    # Accession number (SEC) or series ID (FRED).
    identifier: str = Field(min_length=1, max_length=64)
    entity: str = Field(min_length=1, max_length=300)
    # Taxonomy-qualified XBRL concept or the FRED series title.
    concept: str = Field(min_length=1, max_length=300)
    form: str | None = Field(default=None, max_length=16)
    fiscal_year: int | None = Field(default=None, ge=1900, le=2200)
    fiscal_period: str | None = Field(default=None, max_length=8)
    period_label: str = Field(min_length=1, max_length=40)
    period_start: date | None = None
    period_end: date
    unit: str = Field(min_length=1, max_length=120)
    # Power of ten applied to ``value`` to reach base units (FRED "Billions of Dollars" is 9).
    scale: int = Field(default=0, ge=0, le=12)
    value: Decimal
    # Filing date (SEC) or the vintage date the value first became known (FRED).
    published_at: date
    retrieved_at: datetime
    as_of: date
    source_url: str = Field(min_length=1, max_length=2048)
    freshness: Freshness
    amended: bool = False
    # Filing accession or vintage date that later changed the value (never the new value).
    superseded_by: str | None = Field(default=None, max_length=64)
    revised_before_as_of: bool = False

    @field_validator("source_url")
    @classmethod
    def allowlisted(cls, value: str) -> str:
        if not is_source_url(value):
            raise ValueError("Source URL must be an HTTPS sec.gov or fred.stlouisfed.org link")
        return value

    @field_validator("value")
    @classmethod
    def finite(cls, value: Decimal) -> Decimal:
        if not plausible(value):
            raise ValueError("Value must be finite, below 1e30 and at most 20 decimal places")
        return value
