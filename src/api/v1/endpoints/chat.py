"""
Chat and conversation routes.

Every route depends on `get_current_user`, and every service call is
scoped to `current_user.id` — a user can only ever ask questions against
their own knowledge base and read/delete their own conversations. Thin
controllers only: validation and orchestration live in RAGService /
ConversationService.

STREAMING: ``POST /chat/stream`` streams tokens as SSE events, reusing
the same retrieval + prompt construction as the non-streaming endpoint.
The full assistant response (including sources) is persisted to the
database after the stream completes.

SSE Event Format (POST /chat/stream)
------------------------------------
Content-Type: text/event-stream

Events are newline-delimited JSON prefixed with ``data: ``:

  data: {"type":"token","content":"Hello"}\n
  data: {"type":"token","content":" world."}\n
  data: {"type":"sources","sources":[{"document_id":"...","filename":"doc.pdf","page":1,"chunk_index":0}]}\n

Event types:
  ``token``    — Incremental text chunk from the LLM.  Multiple events
                 are emitted as the model generates output.  Concatenate
                 all ``content`` fields to reconstruct the full answer.

  ``sources``  — Final event carrying source citations.  The ``sources``
                 array is empty when the answer is a refusal (insufficient
                 context).

  ``error``    — Emitted on LLM or retrieval failure.  The stream is
                 closed after this event.  ``content`` contains a
                 human-readable error message.

Frontend usage (JavaScript):

  const es = new EventSource(url);          // or use fetch() + ReadableStream
  const source = new EventSource('/api/v1/chat/stream');
  // For POST with body, use fetch():
  const res = await fetch('/api/v1/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ...' },
    body: JSON.stringify({ message: '...' }),
  });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const text = decoder.decode(value);
    for (const line of text.split('\\n')) {
      if (line.startsWith('data: ')) {
        const event = JSON.parse(line.slice(6));
        if (event.type === 'token') appendToUI(event.content);
        if (event.type === 'sources') showCitations(event.sources);
        if (event.type === 'error') showError(event.content);
      }
    }
  }
"""
import json
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

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
from src.core.exceptions import LLMException
from src.db.session import get_db
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
        conversation_id=result.conversation_id,
    )


def _sse(event: dict) -> str:
    """Format a dict as a single Server-Sent Events frame."""
    return f"data: {json.dumps(event)}\n\n"


@router.post(
    "/chat/stream",
    status_code=status.HTTP_200_OK,
    summary="Stream a RAG answer as Server-Sent Events",
    description=(
        "Same retrieval and prompt construction as POST /chat, but the "
        "LLM output is streamed token-by-token. The final SSE event "
        "carries source citations. The assistant response is persisted "
        "to the database after the stream completes."
    ),
    responses={
        200: {
            "description": "SSE stream of token and source events.",
            "content": {"text/event-stream": {}},
        },
    },
)
async def stream_answer(
    request: ChatRequest,
    raw_request: Request,
    current_user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    async def _event_generator():
        try:
            for event in rag_service.answer_stream(
                request.message,
                user_id=current_user.id,
                conversation_id=request.conversation_id,
            ):
                # If the client disconnected, stop generating.
                if await raw_request.is_disconnected():
                    break
                yield _sse(event)
        except LLMException:
            yield _sse({"type": "error", "content": "LLM generation failed."})
        except Exception:
            yield _sse({"type": "error", "content": "An unexpected error occurred."})
        finally:
            # Streaming can outlive the request-scoped dependency teardown
            # (middleware + StreamingResponse), so close the DB session
            # here as well — this releases its transaction back to the
            # pool on every exit path, including client disconnects.
            db.close()

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
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
