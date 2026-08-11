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
    question: str = Field(
        min_length=1,
        max_length=settings.MAX_QUESTION_LENGTH,
        examples=["What does my onboarding runbook say?"],
    )
    conversation_id: UUID | None = None

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Question cannot be empty.")
        return stripped


class SourceRefSchema(BaseModel):
    document_id: UUID
    filename: str
    page_number: int | None = None
    chunk_index: int
    score: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    conversation_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID
    refused: bool = False
    sources: list[SourceRefSchema] = []


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
