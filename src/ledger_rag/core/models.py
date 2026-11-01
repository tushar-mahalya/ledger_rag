from datetime import date
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from ledger_rag.core.provenance import Provenance, SourceKind


class Profile(StrEnum):
    VOICE = "voice"
    CHAT = "chat"


MAX_SUBJECT_LENGTH = 256
MAX_TENANT_LENGTH = 200
MAX_CLAIM_VALUE_LENGTH = 256
ClaimValue = Annotated[str, StringConstraints(min_length=1, max_length=MAX_CLAIM_VALUE_LENGTH)]


class Principal(BaseModel):
    """Authenticated identity; values are bounded and roles/groups deduplicated in order.

    List counts are bounded where untrusted claims enter (``security.auth``), not
    here, because trusted evaluation principals legitimately hold many groups.
    """

    model_config = ConfigDict(hide_input_in_errors=True)

    subject: str = Field(min_length=1, max_length=MAX_SUBJECT_LENGTH)
    tenant_id: str = Field(min_length=1, max_length=MAX_TENANT_LENGTH)
    roles: list[ClaimValue] = Field(default_factory=list)
    groups: list[ClaimValue] = Field(default_factory=list)

    @field_validator("roles", "groups")
    @classmethod
    def deduplicate(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))


class Document(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    tenant_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=1_000_000, pattern=r"\S")
    source: str = Field(min_length=1, max_length=2048)
    published_at: date
    allowed_groups: list[str] = Field(default_factory=list, max_length=100)


class Chunk(BaseModel):
    id: str
    document_id: str
    tenant_id: str
    title: str
    text: str
    parent_text: str
    source: str
    published_at: date
    allowed_groups: list[str] = Field(default_factory=list)
    ordinal: int
    content_hash: str


class Hit(BaseModel):
    chunk: Chunk
    score: float


# Response bounds. web/src/schemas.ts mirrors them and tests/unit/test_contract.py keeps
# the two in sync, so the server never emits a payload the web client rejects.
MAX_CITATIONS = 12
MAX_CITATION_ID_CHARS = 16
MAX_IDENTIFIER_CHARS = 200
MAX_TITLE_CHARS = 500
MAX_SOURCE_CHARS = 2048
MAX_QUOTE_CHARS = 100_000
MAX_ANSWER_CHARS = 100_000
MAX_SHORT_TEXT_CHARS = 500
MAX_DECIMAL_CHARS = 64
MAX_OPERAND_QUOTE_CHARS = 10_000
MAX_STEPS = 16
MAX_OPERANDS = 64
MAX_CALCULATIONS = 16
MAX_REASONS = 32
MAX_UNSUPPORTED_NUMBERS = 64
MAX_UNSUPPORTED_NUMBER_CHARS = 200
MAX_ACTIVITY_LABEL_CHARS = 160
MAX_PLAN_SUBQUESTIONS = 4
MAX_PLAN_SUBQUESTION_CHARS = 200
MAX_PLAN_ENTITIES = 8
MAX_PLAN_ENTITY_CHARS = 64
# Leave room for SSE framing under the client's 1M-character event and 4MB stream limits.
MAX_RESPONSE_JSON_BYTES = 900_000


def _check_response_size(response: BaseModel) -> None:
    if len(response.model_dump_json().encode("utf-8")) > MAX_RESPONSE_JSON_BYTES:
        raise ValueError(f"Response JSON exceeds {MAX_RESPONSE_JSON_BYTES} UTF-8 bytes")


CitationId = Annotated[str, StringConstraints(max_length=MAX_CITATION_ID_CHARS, pattern=r"^C\d+$")]
Identifier = Annotated[str, StringConstraints(max_length=MAX_IDENTIFIER_CHARS)]
ShortText = Annotated[str, StringConstraints(max_length=MAX_SHORT_TEXT_CHARS)]
DecimalText = Annotated[str, StringConstraints(max_length=MAX_DECIMAL_CHARS)]
Label = Annotated[str, StringConstraints(max_length=64)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
Milliseconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Count = Annotated[int, Field(ge=0)]


class Citation(BaseModel):
    id: CitationId
    document_id: Identifier
    source: str = Field(max_length=MAX_SOURCE_CHARS)
    title: str = Field(max_length=MAX_TITLE_CHARS)
    published_at: date
    quote: str = Field(max_length=MAX_QUOTE_CHARS)
    # External evidence (SEC XBRL, FRED) carries typed provenance; corpus evidence does not.
    source_kind: SourceKind = "corpus"
    provenance: Provenance | None = None

    @model_validator(mode="after")
    def provenance_matches_kind(self) -> Self:
        if (self.source_kind == "corpus") != (self.provenance is None) or (
            self.provenance is not None and self.provenance.source_kind != self.source_kind
        ):
            raise ValueError("External citations need provenance of the same source kind")
        return self


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000, pattern=r"\S")
    profile: Profile = Profile.CHAT
    as_of: date | None = None
    top_k: int = Field(default=5, ge=1, le=10)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def bounded_chat_history(self) -> Self:
        if sum(len(message.content) for message in self.history) > 32000:
            raise ValueError("History exceeds 32000 characters")
        if self.profile == Profile.VOICE and self.history:
            raise ValueError("History is not supported for voice")
        return self


ActivityStage = Literal["planning", "retrieval", "generation", "tool", "validation"]
ActivityTool = Literal[
    "calculator", "search_evidence", "lookup_company_facts", "lookup_macro_series"
]
ActivityOutcome = Literal["ok", "error", "timeout", "unavailable"]
ActivityLabel = Literal[
    "Planning query",
    "Retrieving evidence",
    "Generating answer",
    "Calculating",
    "Searching evidence",
    "Looking up SEC filings",
    "Looking up macro data",
    "Checking tool request",
    "Validating citations",
    "Verifying grounding",
]
# Machine-readable operation identifier; ``stage`` stays coarse for older clients.
ActivityKind = Literal[
    "planning",
    "retrieval",
    "generation",
    "calculator",
    "evidence_search",
    "company_facts_lookup",
    "macro_series_lookup",
    "tool_check",
    "citation_validation",
    "grounding_verification",
]
ACTIVITY_KINDS: dict[ActivityKind, tuple[ActivityStage, ActivityLabel]] = {
    "planning": ("planning", "Planning query"),
    "retrieval": ("retrieval", "Retrieving evidence"),
    "generation": ("generation", "Generating answer"),
    "calculator": ("tool", "Calculating"),
    "evidence_search": ("tool", "Searching evidence"),
    "company_facts_lookup": ("tool", "Looking up SEC filings"),
    "macro_series_lookup": ("tool", "Looking up macro data"),
    "tool_check": ("tool", "Checking tool request"),
    "citation_validation": ("validation", "Validating citations"),
    "grounding_verification": ("validation", "Verifying grounding"),
}
# Kept in sync with planning.analysis.Operation and planning.decisions.Route by tests.
PlanIntent = Literal["lookup", "compare", "ratio", "change", "percent_change", "aggregate"]
PlanRoute = Literal["retrieve", "direct_answer", "clarify", "refuse_out_of_scope"]
PlanText = Annotated[str, StringConstraints(max_length=MAX_PLAN_SUBQUESTION_CHARS)]
PlanEntity = Annotated[str, StringConstraints(max_length=MAX_PLAN_ENTITY_CHARS)]


class PlanSummary(BaseModel):
    """Client-safe plan digest: an explicit allowlist with bounded sizes.

    Never carries prompts, history, decision or analysis confidences, or timings.
    ``subquestions`` holds at most the first four (truncated) retrieval sub-questions;
    ``subquestion_count`` is the full count.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent: PlanIntent
    route: PlanRoute
    subquestion_count: int = Field(ge=0, le=64)
    subquestions: list[PlanText] = Field(default_factory=list, max_length=MAX_PLAN_SUBQUESTIONS)
    companies: list[PlanEntity] = Field(default_factory=list, max_length=MAX_PLAN_ENTITIES)
    periods: list[PlanEntity] = Field(default_factory=list, max_length=MAX_PLAN_ENTITIES)


class Activity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: ActivityStage
    status: Literal["started", "completed"]
    label: ActivityLabel
    tool: ActivityTool | None = None
    # Set on completed external lookups only; other activities keep their wire shape.
    outcome: ActivityOutcome | None = None
    kind: ActivityKind
    # Only on the completed planning activity.
    plan: PlanSummary | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if ACTIVITY_KINDS[self.kind] != (self.stage, self.label):
            raise ValueError("Activity kind does not match its stage and label")
        if self.plan is not None and (self.kind != "planning" or self.status != "completed"):
            raise ValueError("Only a completed planning activity carries a plan")
        if self.outcome is not None and (
            self.status != "completed"
            or self.kind not in {"company_facts_lookup", "macro_series_lookup"}
        ):
            raise ValueError("Only completed external lookups carry an outcome")
        return self


class CalculationOperand(BaseModel):
    """A program input with provenance: an evidence span, a prior step or a constant."""

    kind: Literal["evidence", "step", "constant"]
    value: DecimalText
    citation_id: Identifier | None = None
    quote: str | None = Field(default=None, max_length=MAX_OPERAND_QUOTE_CHARS)
    source_value: ShortText | None = None
    scale: Label | None = None
    unit: Label | None = None
    step: Count | None = None


CalculationKind = Literal["currency", "amount", "ratio", "percent", "boolean"]
UnitKind = Literal["currency", "percent", "ratio", "count", "per_share", "unitless"]
UnitScale = Literal["thousand", "million", "billion", "trillion"]
UnitDisplay = Literal["amount", "percent", "ratio", "boolean"]


class CalculationUnit(BaseModel):
    """Deterministic unit of a result, derived from evidence operands by the program DSL.

    ``kind`` gives the meaning of the base-unit value: ``percent`` values are already
    multiplied by 100, ``ratio`` values are fractions. ``currency`` is an ISO 4217 code,
    or null when the evidence symbol is ambiguous (for example ``C$``). ``scale`` is the
    evidence scale word to display amounts in. ``display`` is a rendering hint: a
    ``ratio`` shown as ``percent`` is multiplied by 100, and ``boolean`` is a yes/no flag.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: UnitKind
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    scale: UnitScale | None = None
    display: UnitDisplay


class CalculationStep(BaseModel):
    index: Count
    op: Label
    operands: list[CalculationOperand] = Field(max_length=MAX_OPERANDS)
    result: DecimalText
    # Additive provenance: ratio/percent/boolean results never support scaled amounts.
    kind: CalculationKind | None = None
    citation_ids: list[Identifier] = Field(default_factory=list, max_length=MAX_OPERANDS)
    evidence_backed: bool = False
    unit: CalculationUnit | None = None


class Calculation(BaseModel):
    """A deterministic evidence-bound program trace; ``result`` is in base units."""

    steps: list[CalculationStep] = Field(max_length=MAX_STEPS)
    result: DecimalText
    citation_ids: list[Identifier] = Field(max_length=MAX_OPERANDS)
    # Unit of ``result``; the last step's unit.
    unit: CalculationUnit | None = None

    def referenced_citations(self) -> set[str]:
        """Every citation id the trace points to, including operand provenance."""
        ids = set(self.citation_ids)
        for step in self.steps:
            ids.update(step.citation_ids)
            ids.update(op.citation_id for op in step.operands if op.citation_id is not None)
        return ids


VerificationStatus = Literal["verified", "unverified", "failed", "skipped"]
# Closed, client-safe reason codes. They never carry draft text, prompts or scores.
VerificationReason = Literal[
    "unsupported_numbers", "verifier_unavailable", "verifier_below_threshold"
]
AbstentionReason = Literal[
    "out_of_scope",
    "no_validated_answer",
    "verification_failed",
    "retrieval_score_below_threshold",
    "citation_coverage_below_threshold",
    "tool_errors_above_threshold",
]


class Verification(BaseModel):
    status: VerificationStatus
    mode: Literal["off", "flag", "enforce"]
    checked_numbers: Count = 0
    # Answer-text mentions; empty when the answer abstained so no draft text is leaked.
    unsupported_numbers: list[
        Annotated[str, StringConstraints(max_length=MAX_UNSUPPORTED_NUMBER_CHARS)]
    ] = Field(default_factory=list, max_length=MAX_UNSUPPORTED_NUMBERS)
    reasons: list[VerificationReason] = Field(default_factory=list, max_length=MAX_REASONS)
    verifier: ShortText | None = None
    verifier_score: FiniteFloat | None = None


class AbstentionFeatures(BaseModel):
    hit_count: Count
    top_score: FiniteFloat | None = None
    mean_score: FiniteFloat | None = None
    citation_coverage: FiniteFloat | None = None
    checked_numbers: Count = 0
    unsupported_number_count: Count = 0
    verification_status: VerificationStatus = "skipped"
    verifier_score: FiniteFloat | None = None
    tool_calls: Count = 0
    tool_errors: Count = 0


class AbstentionDecision(BaseModel):
    """Policy output and inputs; thresholds are uncalibrated until fitted on dev data."""

    abstained: bool
    reasons: list[AbstentionReason] = Field(default_factory=list, max_length=MAX_REASONS)
    features: AbstentionFeatures
    calibrated: bool = False


class Answer(BaseModel):
    model_config = ConfigDict(json_schema_extra={"x-max-json-bytes": MAX_RESPONSE_JSON_BYTES})

    answer: str = Field(max_length=MAX_ANSWER_CHARS)
    citations: list[Citation] = Field(max_length=MAX_CITATIONS)
    abstained: bool
    profile: Profile
    retrieval_ms: Milliseconds
    generation_ms: Milliseconds
    request_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_CHARS)
    model: ShortText
    tool_calls: Count = 0
    calculation: Calculation | None = None
    calculations: list[Calculation] = Field(default_factory=list, max_length=MAX_CALCULATIONS)
    verification: Verification | None = None
    abstention: AbstentionDecision | None = None

    @model_validator(mode="after")
    def internally_consistent(self) -> Self:
        """Evidence references resolve, and an abstention carries no draft or evidence."""
        if self.abstained and (
            self.citations
            or self.calculations
            or self.calculation is not None
            or (self.verification is not None and self.verification.unsupported_numbers)
        ):
            raise ValueError("An abstained answer cannot carry evidence or draft numbers")
        cited = {citation.id for citation in self.citations}
        programs = [*self.calculations, *([self.calculation] if self.calculation else [])]
        if any(not program.referenced_citations() <= cited for program in programs):
            raise ValueError("Calculations reference citations missing from the answer")
        _check_response_size(self)
        return self


class Evidence(BaseModel):
    """The streaming ``evidence`` event: authorized sources retrieved before generation."""

    model_config = ConfigDict(json_schema_extra={"x-max-json-bytes": MAX_RESPONSE_JSON_BYTES})

    request_id: Identifier
    retrieval_ms: Milliseconds
    sources: list[Annotated[str, StringConstraints(max_length=MAX_SOURCE_CHARS)]] = Field(
        max_length=MAX_CITATIONS
    )
    citations: list[Citation] = Field(default_factory=list, max_length=MAX_CITATIONS)

    @model_validator(mode="after")
    def bounded_json(self) -> Self:
        _check_response_size(self)
        return self


class IngestRequest(BaseModel):
    documents: list[Document] = Field(min_length=1, max_length=100)
