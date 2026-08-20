"""
Chat and conversation routes.

Every route depends on `get_current_user`, and every service call is
scoped to `current_user.id` — a user can only ever ask questions against
their own knowledge base and read/delete their own conversations. Thin
controllers only: validation and orchestration live in RAGService /
ConversationService.

STREAMING SEAM: `POST /chat` currently returns the full `ChatResponse`
after the whole answer is generated. A future streaming mode will add an
SSE/JSON-stream variant of this route that streams tokens as they are
produced. The contract that makes that swap safe is already in place:
RAGService.answer() returns the complete `RAGResult` (answer + message
ids + sources) which the streaming route would reuse verbatim once the
LLM finishes — the response schema, message ids, and persisted
retrieval_metadata do not change between streaming and non-streaming.
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
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationResponse,
    MessageListResponse,
    MessageResponse,
    SourceRefSchema,
    SourceSchema,
)
from src.models.user_model import User
from src.services.conversation_service import ConversationService
from src.services.rag_service import RAGService

router = APIRouter(tags=["Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question against your private knowledge base",
    description=(
        "Send a question and receive an answer grounded in the user's "
        "uploaded documents. The response includes source citations "
        "(document ID, filename, page, chunk index) for every chunk that "
        "contributed to the answer. Authentication is required; the "
        "search is automatically scoped to the authenticated user's data."
    ),
    response_description="The generated answer and source citations.",
)
def ask_question(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    result = rag_service.answer(
        request.message,
        user_id=current_user.id,
        conversation_id=request.conversation_id,
    )
    return ChatResponse(
        answer=result.answer,
        sources=[
            SourceSchema(
                document_id=UUID(source.document_id),
                filename=source.filename,
                page=source.page_number,
                chunk_index=source.chunk_index,
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
    "/conversations/{conversation_id}",
    response_model=ConversationDetailResponse,
    summary="Get one of the current user's conversations with its messages",
)
def get_conversation(
    conversation_id: UUID,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationDetailResponse:
    conversation, messages = conversation_service.get_conversation_detail(
        conversation_id, user_id=current_user.id, limit=limit, offset=offset
    )
    return ConversationDetailResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[MessageResponse.model_validate(m) for m in messages],
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
