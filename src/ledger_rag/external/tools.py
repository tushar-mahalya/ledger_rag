"""Chat tool schemas and execution for external lookups.

Results join the turn's citations as ordinary ``C<n>`` evidence (same cap, same calculator
binding, same numeric verification); errors use the chat tool-error contract
``{"error": "..."}`` with sanitized messages.
"""

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ledger_rag.core.models import Citation
from ledger_rag.external.errors import ExternalDataError
from ledger_rag.external.normalize import ExternalFact
from ledger_rag.external.service import ExternalSession, Lookup, Provider

MAX_CITATIONS = 12
ExternalToolName = Literal["lookup_company_facts", "lookup_macro_series"]
TOOL_PROVIDERS: dict[ExternalToolName, Provider] = {
    "lookup_company_facts": "sec",
    "lookup_macro_series": "fred",
}


class CompanyFactsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: str = Field(min_length=1, max_length=200)
    concepts: list[str] = Field(min_length=1, max_length=4)
    periods: list[str] = Field(default_factory=lambda: ["latest"], min_length=1, max_length=4)
    unit: str | None = Field(default=None, min_length=1, max_length=40)

    @model_validator(mode="after")
    def bounded_items(self) -> "CompanyFactsArguments":
        if any(not 0 < len(item) <= 142 for item in self.concepts):
            raise ValueError("each concept must be 1-142 characters")
        if any(not 0 < len(item) <= 20 for item in self.periods):
            raise ValueError("each period must be 1-20 characters")
        return self


class MacroSeriesArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    series_id: str = Field(min_length=1, max_length=40)
    dates: list[date] = Field(default_factory=list, max_length=6)
    start: date | None = None
    end: date | None = None

    @model_validator(mode="after")
    def one_selection(self) -> "MacroSeriesArguments":
        if self.dates and (self.start or self.end):
            raise ValueError("pass either dates or a start/end range, not both")
        if self.end and not self.start:
            raise ValueError("end requires start")
        if self.start and self.end and self.start > self.end:
            raise ValueError("start must not be after end")
        return self


LOOKUP_COMPANY_FACTS = {
    "type": "function",
    "function": {
        "name": "lookup_company_facts",
        "description": (
            "Read-only official SEC EDGAR XBRL data: values a company reported in its "
            "10-K, 10-Q, 20-F, 40-F or 6-K filings, using only filings made on or before the "
            "as-of date. Use when the retrieved documents lack a reported figure. Returns "
            "citable evidence with the filing form, date and period."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "company": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                    "description": "Ticker (AAPL, BRK-B), CIK, or exact registrant name.",
                },
                "concepts": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {"type": "string", "minLength": 1, "maxLength": 142},
                    "description": (
                        "XBRL element names, e.g. InterestExpense, Revenues, NetIncomeLoss, "
                        "Assets, EarningsPerShareDiluted; optionally prefixed us-gaap:, "
                        "ifrs-full: or dei:."
                    ),
                },
                "periods": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {"type": "string", "minLength": 1, "maxLength": 20},
                    "description": (
                        "FY2023 (fiscal year), FY2023Q2 (fiscal quarter), CY2023 or CY2023Q4 "
                        "(calendar frame), YYYY-MM-DD (balance-sheet date) or latest."
                    ),
                },
                "unit": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 40,
                    "description": "Optional unit: USD, USD/shares, shares or pure.",
                },
            },
            "required": ["company", "concepts"],
            "additionalProperties": False,
        },
    },
}

LOOKUP_MACRO_SERIES = {
    "type": "function",
    "function": {
        "name": "lookup_macro_series",
        "description": (
            "Read-only FRED/ALFRED macroeconomic observations exactly as published on or "
            "before the as-of date (vintage-correct), e.g. FEDFUNDS (monthly effective "
            "federal funds rate), DFF (daily), CPIAUCSL, UNRATE, GDP, DGS10. Pass up to six "
            "dates or a start/end range; with neither, returns the latest known observation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "series_id": {"type": "string", "minLength": 1, "maxLength": 40},
                "dates": {
                    "type": "array",
                    "maxItems": 6,
                    "items": {"type": "string", "format": "date"},
                },
                "start": {"type": "string", "format": "date"},
                "end": {"type": "string", "format": "date"},
            },
            "required": ["series_id"],
            "additionalProperties": False,
        },
    },
}

SCHEMAS: dict[ExternalToolName, dict[str, Any]] = {
    "lookup_company_facts": LOOKUP_COMPANY_FACTS,
    "lookup_macro_series": LOOKUP_MACRO_SERIES,
}


def tool_schemas(session: ExternalSession | None) -> list[dict[str, Any]]:
    if session is None:
        return []
    return [
        SCHEMAS[name] for name, provider in TOOL_PROVIDERS.items() if provider in session.providers
    ]


def is_external_tool(name: object) -> bool:
    return name in TOOL_PROVIDERS


def citation_from_fact(fact: ExternalFact, index: int) -> Citation:
    provenance = fact.provenance
    return Citation(
        id=f"C{index}",
        document_id=provenance.evidence_id,
        source=provenance.source_url,
        title=fact.title,
        published_at=provenance.published_at,
        quote=fact.quote,
        source_kind=provenance.source_kind,
        provenance=provenance,
    )


def _invalid(name: str, error: ValidationError) -> str:
    # Pydantic messages are fixed strings; raw input values are never echoed back.
    first = error.errors(include_input=False, include_url=False)[0]
    where = ".".join(str(part) for part in first["loc"]) or "arguments"
    return f"{name} arguments are invalid ({where}: {first['msg']})"


async def run_external_tool(
    name: ExternalToolName,
    arguments: dict[str, Any],
    session: ExternalSession,
    citations: list[Citation],
    evidence_ids: set[str],
) -> dict[str, Any]:
    """Execute one lookup; raises ExternalDataError, returns the tool payload otherwise."""
    if TOOL_PROVIDERS[name] not in session.providers:
        raise ExternalDataError("disabled", "That lookup is not available")
    lookup: Lookup
    try:
        if name == "lookup_company_facts":
            company = CompanyFactsArguments.model_validate(arguments)
        else:
            macro = MacroSeriesArguments.model_validate(arguments)
    except ValidationError as error:
        raise ExternalDataError("invalid_request", _invalid(name, error)) from None
    if len(citations) >= MAX_CITATIONS:
        raise ExternalDataError(
            "budget_exhausted", "The evidence limit is reached; answer with the cited evidence"
        )
    session.spend()
    if name == "lookup_company_facts":
        lookup = await session.data.company_facts(
            company.company,
            company.concepts,
            company.periods,
            as_of=session.as_of,
            unit=company.unit,
        )
    else:
        lookup = await session.data.macro_series(
            macro.series_id,
            as_of=session.as_of,
            start=macro.start,
            end=macro.end,
            dates=macro.dates or None,
        )
    evidence: list[dict[str, Any]] = []
    by_document = {citation.document_id: citation for citation in citations}
    truncated = False
    for fact in lookup.facts:
        evidence_id = fact.provenance.evidence_id
        citation = by_document.get(evidence_id)
        if citation is None:
            if len(citations) >= MAX_CITATIONS:
                truncated = True
                continue
            citation = citation_from_fact(fact, len(citations) + 1)
            citations.append(citation)
            evidence_ids.add(evidence_id)
            by_document[evidence_id] = citation
        evidence.append(external_payload(citation, fact))
    result: dict[str, Any] = {
        "subject": lookup.subject,
        "as_of": session.as_of.isoformat(),
        "evidence": evidence,
    }
    if lookup.missing:
        result["missing"] = [miss.payload() for miss in lookup.missing]
    if truncated:
        result["truncated"] = "evidence limit reached; some values were not added"
    return result


def external_payload(citation: Citation, fact: ExternalFact) -> dict[str, Any]:
    provenance = fact.provenance
    payload: dict[str, Any] = {
        "id": citation.id,
        "title": citation.title,
        "text": citation.quote,
        "published_at": citation.published_at.isoformat(),
        "period": provenance.period_label,
        "source_kind": provenance.source_kind,
        "freshness": provenance.freshness,
    }
    if provenance.superseded_by:
        payload["revised_after_as_of_by"] = provenance.superseded_by
    if provenance.revised_before_as_of:
        payload["restated_before_as_of"] = True
    return payload
