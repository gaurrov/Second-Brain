"""
Request/response DTOs for the chat/RAG module.

These are the ONLY objects that cross the HTTP boundary for the chat and
conversation endpoints. `conversation_id` in `ChatRequest` is optional and
is only ever used to *attach* the exchange to a conversation the caller
already owns - ownership is re-checked in the service layer, never taken
on trust.
"""
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.config import settings


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=settings.MAX_QUESTION_LENGTH,
        description="The user's question about their knowledge base.",
        examples=["What does my onboarding runbook say?"],
    )
    conversation_id: UUID | None = Field(
        default=None,
        description="Optional existing conversation ID to continue. "
        "Omit to start a new conversation.",
        examples=[],
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Message cannot be empty.")
        return stripped


class SourceSchema(BaseModel):
    """A single source citation returned to the API consumer."""

    document_id: UUID = Field(description="The document that produced this chunk.")
    filename: str = Field(description="Original filename of the source document.")
    page: int | None = Field(default=None, description="Page number within the source document, if available.")
    chunk_index: int = Field(description="Index of the chunk within the document.")


class ChatResponse(BaseModel):
    """Response from the RAG chat endpoint."""

    answer: str = Field(description="The assistant's answer generated from the knowledge base.")
    sources: list[SourceSchema] = Field(
        default_factory=list,
        description="Source citations that supported the answer, ordered by relevance.",
    )


class SourceRefSchema(BaseModel):
    """Full source reference stored in retrieval_metadata on persisted messages."""

    document_id: UUID
    filename: str
    page_number: int | None = None
    chunk_index: int
    score: float
    snippet: str


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    total: int
    conversations: list[ConversationResponse]


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: UUID
    role: str
    content: str
    created_at: datetime
    retrieval_metadata: list[SourceRefSchema] | None = None


class MessageListResponse(BaseModel):
    total: int
    messages: list[MessageResponse]


class ConversationDetailResponse(BaseModel):
    """A single conversation plus its messages (the GET /conversation/{id} view)."""

    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse]
