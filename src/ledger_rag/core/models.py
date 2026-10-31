from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


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


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000, pattern=r"\S")
    profile: Profile = Profile.CHAT
    as_of: date | None = None
    top_k: int = Field(default=5, ge=1, le=10)


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
