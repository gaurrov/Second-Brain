"""
Chat and conversation routes.

Every route depends on `get_current_user`, and every service call is
scoped to `current_user.id` — a user can only ever ask questions against
their own knowledge base and read/delete their own conversations. Thin
controllers only: validation and orchestration live in RAGService /
ConversationService.
"""
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status

from src.api.deps import (
    get_conversation_service,
    get_current_user,
    get_rag_service,
)
from src.api.v1.schemas.chat_schema import (
    ChatRequest,
    ChatResponse,
    ConversationListResponse,
    ConversationResponse,
    MessageListResponse,
    MessageResponse,
    SourceRefSchema,
)
from src.models.user_model import User
from src.services.conversation_service import ConversationService
from src.services.rag_service import RAGService

router = APIRouter(tags=["Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question against your private knowledge base",
)
def ask_question(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    result = rag_service.answer(
        request.question,
        user_id=current_user.id,
        conversation_id=request.conversation_id,
    )
    return ChatResponse(
        answer=result.answer,
        conversation_id=result.conversation_id,
        user_message_id=result.user_message_id,
        assistant_message_id=result.assistant_message_id,
        refused=result.refused,
        sources=[
            SourceRefSchema(
                document_id=UUID(source.document_id),
                filename=source.filename,
                page_number=source.page_number,
                chunk_index=source.chunk_index,
                score=source.score,
                snippet=source.snippet,
            )
            for source in result.sources
        ],
    )


@router.get(
    "/conversations",
    response_model=ConversationListResponse,
    summary="List the current user's conversations",
)
def list_conversations(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationListResponse:
    conversations, total = conversation_service.list_conversations(
        user_id=current_user.id, limit=limit, offset=offset
    )
    return ConversationListResponse(
        total=total,
        conversations=[ConversationResponse.model_validate(c) for c in conversations],
    )


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=MessageListResponse,
    summary="List messages in one of the current user's conversations",
)
def get_conversation_messages(
    conversation_id: UUID,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> MessageListResponse:
    messages = conversation_service.list_messages(
        conversation_id, user_id=current_user.id, limit=limit, offset=offset
    )
    return MessageListResponse(
        total=len(messages),
        messages=[MessageResponse.model_validate(m) for m in messages],
    )


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete one of the current user's conversations",
)
def delete_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> None:
    conversation_service.delete_conversation(conversation_id, user_id=current_user.id)
