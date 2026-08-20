"""
Application configuration.

All environment-driven configuration is centralized here using Pydantic
Settings. Nothing else in the codebase should call os.environ / os.getenv
directly — always import `settings` from this module instead.

Every setting is validated at load time (`field_validator` /
`model_validator`) so a misconfigured environment fails fast on startup
instead of producing confusing runtime failures in production.

Production-only validators (secret strength, required GROQ key) only fire
when ``APP_ENV=production`` so local development / CI stay unblocked.
"""
import logging
from functools import lru_cache
from typing import Any, Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_APP_ENVIRONMENTS = {"development", "staging", "production", "test"}
_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_ALLOWED_DB_SCHEMES = {"postgresql", "postgresql+psycopg2", "sqlite", "sqlite+aiosqlite"}
_ALLOWED_REDIS_SCHEMES = {"redis", "rediss"}


class Settings(BaseSettings):
    # --- Application ---
    APP_NAME: str = "Second Brain API"
    APP_ENV: str = "development"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "json"

    # --- PostgreSQL ---
    POSTGRES_USER: str = "secondbrain"
    POSTGRES_PASSWORD: str = "change_me"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "secondbrain_db"
    DATABASE_URL: str | None = None

    # Connection pooling. Tuned per Postgres-style servers; these values
    # are clamped to sane ranges by validators.
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT_SECONDS: int = 30
    DB_POOL_RECYCLE_SECONDS: int = 1800
    DB_ECHO: bool = False
    # Queries slower than this (ms) are logged as warnings for spotting N+1s
    # and missing indexes.
    DB_SLOW_QUERY_THRESHOLD_MS: int = 200

    # --- JWT / Security ---
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    MAX_FAILED_LOGIN_ATTEMPTS: int = 5
    ACCOUNT_LOCK_MINUTES: int = 15

    # --- CORS ---
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # --- Request IDs ---
    # Whether to honor an incoming X-Request-ID header (useful for
    # correlating a user-facing trace with our logs) or always mint a
    # fresh UUID. If the incoming header is empty/invalid it is ignored.
    PROPAGATE_REQUEST_ID: bool = True
    REQUEST_ID_HEADER: str = "X-Request-ID"

    # --- Structured logging / metrics ---
    METRICS_ENABLED: bool = True
    METRICS_PATH: str = "/metrics"

    # --- Redis ---
    REDIS_ENABLED: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_PREFIX: str = "secondbrain"
    REDIS_SOCKET_TIMEOUT_SECONDS: float = 1.0
    REDIS_CONNECTION_POOL_SIZE: int = 20
    # Default TTL (seconds) applied to cache entries that don't set one.
    REDIS_DEFAULT_TTL_SECONDS: int = 300

    # --- Application-level caching (Redis-backed, degrade to no-op) ---
    CACHE_ENABLED: bool = False
    CACHE_TTL_SECONDS: int = 300

    # --- Rate limiting ---
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_DEFAULT_LIMIT: int = 120
    RATE_LIMIT_DEFAULT_WINDOW_SECONDS: int = 60
    RATE_LIMIT_LOGIN_LIMIT: int = 10
    RATE_LIMIT_LOGIN_WINDOW_SECONDS: int = 60
    RATE_LIMIT_STRATEGY: Literal["fixed_window"] = "fixed_window"
    # Headers (e.g. "X-Forwarded-For") used to derive the client identity
    # when running behind a trusted reverse proxy. Only the first value is
    # used. Empty by default => falls back to the direct peer address.
    RATE_LIMIT_TRUSTED_HEADERS: list[str] = []

    # --- Retry / resilience ---
    RETRY_MAX_ATTEMPTS: int = 3
    RETRY_BACKOFF_BASE_SECONDS: float = 0.2
    RETRY_MAX_DELAY_SECONDS: float = 5.0
    RETRY_JITTER: bool = True

    # --- File storage ---
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE_MB: int = 25

    # --- Qdrant ---
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_USE_HTTPS: bool = False
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "documents_kb"
    # Points per upsert request. Large documents are chunked into batches
    # of this size so a single request never carries the whole document.
    QDRANT_UPSERT_BATCH_SIZE: int = 256
    # Per-call timeout for Qdrant REST/gRPC operations.
    QDRANT_TIMEOUT_SECONDS: float = 10.0
    # Use the Qdrant gRPC transport when available (lower overhead / better
    # throughput for large upserts). Requires QDRANT_GRPC_PORT to be open.
    QDRANT_GRPC_ENABLED: bool = False
    # How long app startup is willing to wait for Qdrant before continuing
    # (ingestion surfaces a clear error later if Qdrant is still down).
    STARTUP_QDRANT_TIMEOUT_SECONDS: float = 5.0

    # --- Embeddings ---
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIMENSION: int = 768  # must match the model above
    EMBEDDING_BATCH_SIZE: int = 32
    # Max unique texts held in the text -> vector LRU cache. Bounded so
    # cached embeddings can't grow unbounded and eat RAM in long-running
    # workers.
    EMBEDDING_CACHE_SIZE: int = 4096
    # Optional torch device ("cpu", "cuda", "mps"). None => let the model
    # pick its default (usually CPU).
    EMBEDDING_DEVICE: str | None = None
    # Optional L2 (Redis) embedding cache shared across workers/instances.
    # Text -> vector JSON blobs with a long TTL since embeddings are stable.
    EMBEDDING_REDIS_CACHE_ENABLED: bool = False
    EMBEDDING_REDIS_CACHE_TTL_SECONDS: int = 86400

    # --- Chunking ---
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120

    # --- Groq (LLM generation) ---
    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_MAX_TOKENS: int = 1024
    GROQ_TEMPERATURE: float = 0.2
    GROQ_TIMEOUT_SECONDS: int = 60

    # --- RAG retrieval ---
    RETRIEVAL_TOP_K: int = 8
    RETRIEVAL_SCORE_THRESHOLD: float = 0.30
    RERANK_ENABLED: bool = False
    RERANK_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_K: int = 4

    # --- Context compression ---
    CONTEXT_MAX_CHARACTERS: int = 6000
    CONTEXT_DEDUPE_THRESHOLD: float = 0.95

    # --- Conversations ---
    CONVERSATION_HISTORY_LIMIT: int = 8
    # Approximate character budget for conversation history included in
    # the LLM prompt.  History is first capped by message count, then
    # trimmed to this character budget (most recent messages kept).  A
    # rough 4-chars-per-token ratio is used; set to 0 to disable.
    CONVERSATION_HISTORY_MAX_CHARACTERS: int = 4000
    MAX_QUESTION_LENGTH: int = 2000

    # --- RAG answer cache (Redis-backed, optional) ---
    # Short-TTL cache of (user, question) -> answer+sources. Disabled by
    # default because identical questions asked in different conversations
    # get identical answers for the TTL window.
    RAG_CACHE_ENABLED: bool = False
    RAG_CACHE_TTL_SECONDS: int = 300

    # --- Background workers ---
    # "background": FastAPI BackgroundTasks (in-process, per-request).
    # "pool": a dedicated thread pool shared across requests (survives
    #         request completion, but not process restarts).
    # "rq": Redis Queue — persistent, retry-capable, monitored.
    TASK_WORKER: Literal["background", "pool", "rq"] = "background"
    WORKER_CONCURRENCY: int = 2
    WORKER_QUEUE_MAXSIZE: int = 1000
    # Seconds the pool waits for in-flight tasks on shutdown before
    # cancelling.
    WORKER_SHUTDOWN_TIMEOUT_SECONDS: int = 30
    # RQ-specific settings (only used when TASK_WORKER=rq).
    RQ_QUEUE_NAME: str = "secondbrain"
    RQ_DEFAULT_TIMEOUT: int = 300  # seconds per job
    RQ_MAX_RETRIES: int = 3
    RQ_RESULT_TTL: int = 3600  # how long results stay in Redis

    # --- Startup performance ---
    # Preload the embedding model during app startup (first request after
    # deploy pays a multi-second load otherwise). Costs memory at boot.
    STARTUP_WARMUP: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("APP_ENV")
    @classmethod
    def _validate_app_env(cls, value: str) -> str:
        value = value.lower()
        if value not in _APP_ENVIRONMENTS:
            raise ValueError(
                f"APP_ENV must be one of {sorted(_APP_ENVIRONMENTS)}, got '{value}'."
            )
        return value

    @field_validator("LOG_LEVEL")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        value = value.upper()
        if value not in _LOG_LEVELS:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(_LOG_LEVELS)}, got '{value}'.")
        return value

    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def _validate_jwt_secret(cls, value: str, info: Any) -> str:
        env = info.data.get("APP_ENV", "development")
        if env == "production":
            if len(value) < 32:
                raise ValueError(
                    "JWT_SECRET_KEY must be at least 32 characters in production. "
                    "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
                )
            if value in ("CHANGE_THIS_TO_A_LONG_RANDOM_SECRET", "change_me", "test-secret-key-for-ci-only"):
                raise ValueError("JWT_SECRET_KEY must not use a known default value in production.")
        return value

    @field_validator("DATABASE_URL")
    @classmethod
    def _validate_database_url(cls, value: str | None) -> str | None:
        if not value:
            return None
        scheme = value.split("://", 1)[0]
        if scheme not in _ALLOWED_DB_SCHEMES:
            raise ValueError(
                f"DATABASE_URL scheme '{scheme}' is not supported. "
                f"Allowed: {sorted(_ALLOWED_DB_SCHEMES)}."
            )
        return value

    @field_validator("REDIS_URL")
    @classmethod
    def _validate_redis_url(cls, value: str) -> str:
        scheme = value.split("://", 1)[0]
        if scheme not in _ALLOWED_REDIS_SCHEMES:
            raise ValueError(
                f"REDIS_URL scheme '{scheme}' is not supported. "
                f"Allowed: {sorted(_ALLOWED_REDIS_SCHEMES)}."
            )
        return value

    @field_validator("POSTGRES_PORT", "QDRANT_PORT", "QDRANT_GRPC_PORT")
    @classmethod
    def _validate_ports(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError(f"Port must be in 1..65535, got {value}.")
        return value

    @field_validator("EMBEDDING_DIMENSION")
    @classmethod
    def _validate_embedding_dimension(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"EMBEDDING_DIMENSION must be positive, got {value}.")
        return value

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def _validate_chunk_overlap(cls, value: int, info: Any) -> int:
        chunk_size = info.data.get("CHUNK_SIZE", 800)
        if value < 0 or value >= chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP must be >= 0 and < CHUNK_SIZE ({chunk_size}), got {value}."
            )
        return value

    @field_validator("RETRIEVAL_SCORE_THRESHOLD")
    @classmethod
    def _validate_score_threshold(cls, value: float) -> float:
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"RETRIEVAL_SCORE_THRESHOLD must be in [-1, 1], got {value}.")
        return value

    @field_validator("RETRY_MAX_ATTEMPTS", "WORKER_CONCURRENCY", "WORKER_QUEUE_MAXSIZE")
    @classmethod
    def _validate_positive_ints(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"Value must be >= 1, got {value}.")
        return value

    @field_validator("RETRY_BACKOFF_BASE_SECONDS", "RETRY_MAX_DELAY_SECONDS")
    @classmethod
    def _validate_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError(f"Value must be >= 0, got {value}.")
        return value

    @model_validator(mode="after")
    def _validate_cross_field_constraints(self) -> "Settings":
        if self.APP_ENV == "production":
            if not self.GROQ_API_KEY:
                raise ValueError(
                    "GROQ_API_KEY is required when APP_ENV=production (RAG answering needs an LLM)."
                )
        if self.REDIS_ENABLED and self.RATE_LIMIT_ENABLED is False and self.CACHE_ENABLED is False:
            # Redis is only useful when something consumes it; this is a
            # hint, not an error — keep it as a passive check so a typo'd
            # flag surfaces early.
            pass
        if self.REDIS_SOCKET_TIMEOUT_SECONDS <= 0:
            raise ValueError("REDIS_SOCKET_TIMEOUT_SECONDS must be > 0.")
        return self

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def sqlalchemy_database_uri(self) -> str:
        """
        Build the SQLAlchemy connection string.
        If DATABASE_URL is explicitly set (e.g. in production / Docker),
        it always takes precedence over the composed value.
        """
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def log_level_int(self) -> int:
        """Numeric logging level for stdlib logging."""
        return getattr(logging, self.LOG_LEVEL)

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings accessor. Using lru_cache ensures the .env file
    and environment are parsed only once per process.
    """
    return Settings()


settings = get_settings()
