"""
Shared FastAPI dependencies.

`get_current_user` is the cornerstone of multi-user isolation: it is the
ONLY place in the codebase that derives a user's identity from a request.
Every endpoint that touches user-owned data must depend on it (directly
or transitively) rather than trusting a user_id from the request body,
query params, or path.
"""
from uuid import UUID

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from src.core.exceptions import InactiveUserException, InvalidTokenException
from src.core.security import TokenType, decode_token
from src.db.session import get_db
from src.models.user_model import User
from src.repositories.document_repository import DocumentRepository
from src.repositories.user_repository import UserRepository
from src.services.auth_service import AuthService
from src.services.document_service import DocumentService
from src.services.ingestion_service import process_document_task

# tokenUrl is documentation-only here (points Swagger's "Authorize" button
# at the login endpoint); the actual login route returns JSON, not an
# OAuth2 form redirect.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")


def get_user_repository(db: Session = Depends(get_db)) -> UserRepository:
    return UserRepository(db)


def get_auth_service(
    user_repository: UserRepository = Depends(get_user_repository),
) -> AuthService:
    return AuthService(user_repository)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    user_repository: UserRepository = Depends(get_user_repository),
) -> User:
    """
    Decode the bearer access token, look up the user it belongs to, and
    return it. Raises InvalidTokenException / InactiveUserException on
    any failure — handled centrally by the exception handlers in main.py.
    """
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


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Alias kept explicit for readability at call sites that care about activity status."""
    return current_user


def get_document_repository(db: Session = Depends(get_db)) -> DocumentRepository:
    return DocumentRepository(db)


def get_document_service(
    background_tasks: BackgroundTasks,
    document_repository: DocumentRepository = Depends(get_document_repository),
    db: Session = Depends(get_db),
) -> DocumentService:
    """
    Wires DocumentService with a background-task dispatcher for
    processing. `process_document_task` opens its own DB session (see
    ingestion_service.py) rather than reusing the request's session,
    since the request session is closed before background tasks run.
    Swapping to Celery/RQ later means changing only the body of
    `dispatch_processing` (call `.delay(document_id)` instead of
    `background_tasks.add_task`) — no change needed in DocumentService
    or the ingestion pipeline itself.
    """

    def dispatch_processing(document_id: UUID) -> None:
        background_tasks.add_task(process_document_task, document_id)

    return DocumentService(
        document_repository=document_repository,
        db=db,
        dispatch_processing=dispatch_processing,
    )
