"""
FastAPI application entrypoint.

Responsibilities of this module ONLY:
  - instantiate the FastAPI app
  - register middleware (CORS, request logging, etc.)
  - register global exception handlers (domain exceptions -> HTTP responses)
  - mount the versioned API router
  - expose a lightweight health check

No business logic, no route handlers, and no DB queries belong here.
"""
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.v1.router import api_router
from src.core.config import settings
from src.core.exceptions import (
    AppException,
    ConversationNotFoundException,
    DocumentNotFoundException,
    EmptyFileException,
    FileTooLargeException,
    InactiveUserException,
    InvalidCredentialsException,
    InvalidTokenException,
    LLMException,
    PromptInjectionException,
    TokenExpiredException,
    UnsupportedFileTypeException,
    UserAlreadyExistsException,
    UserNotFoundException,
)

logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("second_brain")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown lifecycle."""
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    try:
        from src.vectorstore.collection_manager import ensure_collection
        from src.vectorstore.qdrant_client import get_qdrant_client

        ensure_collection(get_qdrant_client())
    except Exception as exc:
        # Don't crash app startup if Qdrant isn't reachable yet (e.g.
        # local dev before `docker compose up`); ingestion calls will
        # surface a clear error instead when a document is uploaded.
        logger.warning("Could not initialize Qdrant collection on startup: %s", exc)

    yield


def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description="Backend API for a multi-user Personal Knowledge Management (Second Brain) platform.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    _register_middleware(app)
    _register_exception_handlers(app)
    _register_routers(app)

    return app


def _register_middleware(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        """
        Attaches a correlation/request ID to every request for tracing
        across logs, and logs basic timing/status information.
        """
        request_id = str(uuid4())
        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


def _register_exception_handlers(app: FastAPI) -> None:
    """
    Maps domain exceptions (src.core.exceptions) to appropriate HTTP
    status codes. This keeps the service/repository layers free of any
    knowledge of HTTP status codes.
    """

    @app.exception_handler(UserAlreadyExistsException)
    async def user_exists_handler(request: Request, exc: UserAlreadyExistsException):
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": exc.message})

    @app.exception_handler(InvalidCredentialsException)
    async def invalid_credentials_handler(request: Request, exc: InvalidCredentialsException):
        return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": exc.message})

    @app.exception_handler(InvalidTokenException)
    async def invalid_token_handler(request: Request, exc: InvalidTokenException):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": exc.message},
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(TokenExpiredException)
    async def token_expired_handler(request: Request, exc: TokenExpiredException):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": exc.message},
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(InactiveUserException)
    async def inactive_user_handler(request: Request, exc: InactiveUserException):
        return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content={"detail": exc.message})

    @app.exception_handler(UserNotFoundException)
    async def user_not_found_handler(request: Request, exc: UserNotFoundException):
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": exc.message})

    @app.exception_handler(DocumentNotFoundException)
    async def document_not_found_handler(request: Request, exc: DocumentNotFoundException):
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": exc.message})

    @app.exception_handler(ConversationNotFoundException)
    async def conversation_not_found_handler(request: Request, exc: ConversationNotFoundException):
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": exc.message})

    @app.exception_handler(PromptInjectionException)
    async def prompt_injection_handler(request: Request, exc: PromptInjectionException):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": exc.message})

    @app.exception_handler(LLMException)
    async def llm_exception_handler(request: Request, exc: LLMException):
        logger.warning("LLM generation failed: %s", exc.message)
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY, content={"detail": exc.message}
        )

    @app.exception_handler(UnsupportedFileTypeException)
    async def unsupported_file_type_handler(request: Request, exc: UnsupportedFileTypeException):
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, content={"detail": exc.message}
        )

    @app.exception_handler(FileTooLargeException)
    async def file_too_large_handler(request: Request, exc: FileTooLargeException):
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, content={"detail": exc.message}
        )

    @app.exception_handler(EmptyFileException)
    async def empty_file_handler(request: Request, exc: EmptyFileException):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": exc.message})

    @app.exception_handler(AppException)
    async def generic_app_exception_handler(request: Request, exc: AppException):
        # Catch-all fallback for any future AppException subclass that
        # doesn't have a dedicated handler above.
        logger.warning("Unhandled AppException: %s", exc.message)
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": exc.message})


def _register_routers(app: FastAPI) -> None:
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    @app.get("/health", tags=["Health"], summary="Liveness/readiness check")
    def health_check():
        return {"status": "ok", "app": settings.APP_NAME, "env": settings.APP_ENV}


app = create_application()
