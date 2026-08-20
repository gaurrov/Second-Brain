"""
FastAPI application entrypoint.

Responsibilities of this module ONLY:
  - configure structured logging
  - instantiate the FastAPI app
  - register middleware (CORS, request ID + access logs, metrics, rate limit)
  - register global exception handlers (domain exceptions -> HTTP responses)
  - mount the versioned API router
  - expose health (liveness + readiness) and Prometheus metrics endpoints
  - own the application lifecycle (startup/shutdown of connections, pools)

No business logic, no route handlers, and no DB queries belong here.
"""
import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.v1.router import api_router
from src.core.config import settings
from src.core.context import reset_request_id, set_request_id
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
from src.core.health import get_health_checker
from src.core.logging import setup_logging
from src.core.metrics import http_requests_in_flight, metrics_payload, normalize_path, record_request
from src.core.rate_limiter import get_rate_limiter

setup_logging()
logger = logging.getLogger("second_brain")

_RATE_LIMIT_EXEMPT_PATHS = {"/health", "/health/live", "/health/ready", "/metrics", "/docs", "/redoc", "/openapi.json"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown lifecycle."""
    logger.info(
        "Starting %s (env=%s log_format=%s)",
        settings.APP_NAME, settings.APP_ENV, settings.LOG_FORMAT,
    )
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    # --- Startup: warm shared infrastructure (never fatal) ---
    await _init_qdrant_collection()

    if settings.REDIS_ENABLED:
        from src.core.redis_client import get_redis

        get_redis()  # construct the pool now; failures surface on first use

    if settings.TASK_WORKER == "pool":
        from src.workers.pool import get_worker_pool

        get_worker_pool().start()
    elif settings.TASK_WORKER == "rq":
        logger.info("Task dispatcher: RQ (separate worker process)")

    if settings.STARTUP_WARMUP:
        _start_model_warmup()

    yield

    # --- Shutdown: drain work, then close connections ---
    logger.info("Shutting down %s", settings.APP_NAME)
    if settings.TASK_WORKER == "pool":
        from src.workers.pool import stop_worker_pool

        stop_worker_pool()
    # RQ workers are separate processes; nothing to shut down here.

    if settings.REDIS_ENABLED:
        from src.core.redis_client import close_redis

        close_redis()

    from src.db.session import dispose_engine
    from src.vectorstore.qdrant_client import close_qdrant_client

    dispose_engine()
    close_qdrant_client()
    logger.info("Shutdown complete")


async def _init_qdrant_collection() -> None:
    """Best-effort collection provisioning, bounded by a startup timeout."""
    try:
        from src.vectorstore.collection_manager import ensure_collection
        from src.vectorstore.qdrant_client import get_qdrant_client

        await asyncio.wait_for(
            asyncio.to_thread(ensure_collection, get_qdrant_client()),
            timeout=settings.STARTUP_QDRANT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Qdrant collection provisioning timed out after %.1fs; "
            "ingestion will surface a clear error instead.",
            settings.STARTUP_QDRANT_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 - don't crash startup if Qdrant is down
        logger.warning("Could not initialize Qdrant collection on startup: %s", exc)


def _start_model_warmup() -> None:
    """Preload the embedding model in a background thread (STARTUP_WARMUP)."""
    def _load() -> None:
        try:
            from src.services.embedding_service import _get_model

            _get_model()
            logger.info("Embedding model preloaded (STARTUP_WARMUP)")
        except Exception:  # noqa: BLE001 - warmup is best-effort
            logger.warning("Embedding model preload failed", exc_info=True)

    threading.Thread(target=_load, name="sb-warmup", daemon=True).start()


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
    _register_health_and_metrics(app)

    return app


def _register_middleware(app: FastAPI) -> None:
    """
    Middleware order (outermost -> innermost):
        CORS -> request ID/access log -> metrics -> rate limit -> routes

    Note: `add_middleware` prepends, so registration below is written in
    reverse (rate limit first, CORS last).
    """
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        if not settings.RATE_LIMIT_ENABLED or request.method == "OPTIONS":
            return await call_next(request)
        path = request.url.path
        if path in _RATE_LIMIT_EXEMPT_PATHS:
            return await call_next(request)

        client_id = _client_identity(request)
        is_login = path.startswith(f"{settings.API_V1_PREFIX}/auth/")
        if not get_rate_limiter().check(client_id, login=is_login):
            retry_after = (
                settings.RATE_LIMIT_LOGIN_WINDOW_SECONDS if is_login
                else settings.RATE_LIMIT_DEFAULT_WINDOW_SECONDS
            )
            logger.warning(
                "Rate limit exceeded for client=%s path=%s", client_id, path,
                extra={"client": client_id, "path": path},
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Too many requests. Please slow down and try again."},
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        if not settings.METRICS_ENABLED:
            return await call_next(request)
        method = request.method
        label_path = normalize_path(request.url.path)
        http_requests_in_flight.labels(method=method, path=label_path).inc()
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            record_request(method, request.url.path, 500, time.perf_counter() - start)
            raise
        finally:
            http_requests_in_flight.labels(method=method, path=label_path).dec()
        record_request(method, request.url.path, response.status_code, time.perf_counter() - start)
        return response

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        """
        Attaches a correlation/request ID to every request for tracing
        across logs (structured JSON with `request_id`), honors an
        incoming header when PROPAGATE_REQUEST_ID is set, and logs basic
        timing/status information for every request.
        """
        request_id = _resolve_request_id(request)
        token = set_request_id(request_id)
        start_time = time.perf_counter()
        status_code = 500
        response: Response | None = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            reset_request_id(token)
            duration_ms = (time.perf_counter() - start_time) * 1000
            if response is not None:
                response.headers[settings.REQUEST_ID_HEADER] = request_id
            logger.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            )

    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )


def _resolve_request_id(request: Request) -> str:
    if settings.PROPAGATE_REQUEST_ID:
        incoming = request.headers.get(settings.REQUEST_ID_HEADER)
        if incoming and incoming.strip() and len(incoming) <= 128:
            return incoming.strip()
    return str(uuid4())


def _client_identity(request: Request) -> str:
    """Best-effort client identity for rate limiting."""
    for header_name in settings.RATE_LIMIT_TRUSTED_HEADERS:
        value = request.headers.get(header_name)
        if value:
            return value.split(",", 1)[0].strip()
    if request.client is not None:
        return request.client.host
    return "unknown"


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
        logger.debug("LLM generation failed: %s", exc.message)
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": "The language model could not generate a response. Please try again."},
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
        logger.debug("Unhandled AppException: %s", exc.message)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "An error occurred processing your request."},
        )


def _register_routers(app: FastAPI) -> None:
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)


def _register_health_and_metrics(app: FastAPI) -> None:
    checker = get_health_checker()

    @app.get("/health", tags=["Health"], summary="Liveness check")
    def health_check() -> dict:
        return checker.check_live()

    @app.get("/health/live", tags=["Health"], summary="Liveness check")
    def health_live() -> dict:
        return checker.check_live()

    @app.get("/health/ready", tags=["Health"], summary="Readiness check (DB, Qdrant, Redis)")
    def health_ready(response: Response) -> dict:
        report, ready = checker.check_ready()
        if not ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return report

    @app.get(settings.METRICS_PATH, tags=["Observability"], summary="Prometheus metrics", include_in_schema=False)
    def metrics() -> Response:
        content_type, body = metrics_payload()
        return Response(content=body, media_type=content_type)


app = create_application()
