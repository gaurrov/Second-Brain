"""
Shared FastAPI dependencies.

`get_current_user` is the cornerstone of multi-user isolation: it is the
ONLY place in the codebase that derives a user's identity from a request.
Every endpoint that touches user-owned data must depend on it (directly
or transitively) rather than trusting a user_id from the request body,
query params, or path.
"""
from uuid import UUID

from fastapi import BackgroundTasks, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from src.core.config import settings
from src.core.exceptions import InactiveUserException, InvalidTokenException
from src.core.security import TokenType, decode_token
from src.db.session import get_db
from src.models.user_model import User
from src.repositories.conversation_repository import ConversationRepository
from src.repositories.document_repository import DocumentRepository
from src.repositories.message_repository import MessageRepository
from src.repositories.refresh_token_repository import RefreshTokenRepository
from src.repositories.user_repository import UserRepository
from src.services.auth_service import AuthService
from src.services.conversation_service import ConversationService
from src.services.document_service import DocumentService
from src.services.ingestion_service import process_document_task
from src.services.rag_service import RAGService, build_rag_service

bearer_scheme = HTTPBearer(auto_error=False)


def get_user_repository(db: Session = Depends(get_db)) -> UserRepository:
    return UserRepository(db)


def get_refresh_token_repository(db: Session = Depends(get_db)) -> RefreshTokenRepository:
    return RefreshTokenRepository(db)


def get_auth_service(
    user_repository: UserRepository = Depends(get_user_repository),
    refresh_token_repository: RefreshTokenRepository = Depends(get_refresh_token_repository),
) -> AuthService:
    return AuthService(user_repository, refresh_token_repository)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    user_repository: UserRepository = Depends(get_user_repository),
) -> User:
    """
    Decode the bearer access token, look up the user it belongs to, and
    return it. Raises InvalidTokenException / InactiveUserException on
    any failure — handled centrally by the exception handlers in main.py.
    """
    if credentials is None:
        raise InvalidTokenException("Missing bearer token.")
    token = credentials.credentials
    payload = decode_token(token, expected_type=TokenType.ACCESS)

    try:
        user_id = UUID(payload["sub"])
    except (ValueError, TypeError):
        raise InvalidTokenException()

    user = user_repository.get_by_id(user_id)
    if user is None:
        raise InvalidTokenException()
    if not user.is_active:
        raise InactiveUserException()

    return user


def get_document_repository(db: Session = Depends(get_db)) -> DocumentRepository:
    return DocumentRepository(db)


def get_document_service(
    background_tasks: BackgroundTasks,
    document_repository: DocumentRepository = Depends(get_document_repository),
    db: Session = Depends(get_db),
) -> DocumentService:
    """
    Wires DocumentService with a background-task dispatcher for
    processing. ``process_document_task`` opens its own DB session (see
    ingestion_service.py) rather than reusing the request's session,
    since the request session is closed before background tasks run.

    Supported dispatchers (``TASK_WORKER`` setting):
      - "background": FastAPI BackgroundTasks (per-request, in-process)
      - "pool":       in-process thread pool (survives request completion)
      - "rq":         Redis Queue (persistent, retry-capable, monitored)
    """
    if settings.TASK_WORKER == "pool":
        from src.workers.pool import get_worker_pool

        def dispatch_processing(document_id: UUID) -> None:
            get_worker_pool().submit(process_document_task, document_id)

    elif settings.TASK_WORKER == "rq":
        from src.workers.rq_worker import dispatch_to_rq

        dispatch_processing = dispatch_to_rq  # type: ignore[assignment]

    else:

        def dispatch_processing(document_id: UUID) -> None:
            background_tasks.add_task(process_document_task, document_id)

    return DocumentService(
        document_repository=document_repository,
        db=db,
        dispatch_processing=dispatch_processing,
    )


def get_conversation_repository(db: Session = Depends(get_db)) -> ConversationRepository:
    return ConversationRepository(db)


def get_message_repository(db: Session = Depends(get_db)) -> MessageRepository:
    return MessageRepository(db)


def get_conversation_service(
    conversation_repository: ConversationRepository = Depends(get_conversation_repository),
    message_repository: MessageRepository = Depends(get_message_repository),
) -> ConversationService:
    return ConversationService(conversation_repository, message_repository)


def get_rag_service(db: Session = Depends(get_db)) -> RAGService:
    """Wire the full RAG pipeline for the request's DB session."""
    return build_rag_service(db)
