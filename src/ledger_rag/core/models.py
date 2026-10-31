from datetime import date
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Profile(StrEnum):
    VOICE = "voice"
    CHAT = "chat"


class Principal(BaseModel):
    subject: str
    tenant_id: str
    roles: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)


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


class Citation(BaseModel):
    id: str
    document_id: str
    source: str
    title: str
    published_at: date
    quote: str


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
ActivityTool = Literal["calculator", "search_evidence"]
ActivityLabel = Literal[
    "Planning query",
    "Retrieving evidence",
    "Generating answer",
    "Calculating",
    "Searching evidence",
    "Checking tool request",
    "Validating citations",
]


class Activity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: ActivityStage
    status: Literal["started", "completed"]
    label: ActivityLabel
    tool: ActivityTool | None = None


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    abstained: bool
    profile: Profile
    retrieval_ms: float
    generation_ms: float
    request_id: str
    model: str
    tool_calls: int = 0


class IngestRequest(BaseModel):
    documents: list[Document] = Field(min_length=1, max_length=100)
