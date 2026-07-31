"""
Application configuration.

All environment-driven configuration is centralized here using Pydantic
Settings. Nothing else in the codebase should call os.environ / os.getenv
directly — always import `settings` from this module instead.
"""
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Application ---
    APP_NAME: str = "Second Brain API"
    APP_ENV: str = "development"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"

    # --- PostgreSQL ---
    POSTGRES_USER: str = "secondbrain"
    POSTGRES_PASSWORD: str = "change_me"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "secondbrain_db"
    DATABASE_URL: str | None = None

    # --- JWT / Security ---
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    MAX_FAILED_LOGIN_ATTEMPTS: int = 5
    ACCOUNT_LOCK_MINUTES: int = 15

    # --- CORS ---
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    # --- File storage ---
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE_MB: int = 25

    # --- Qdrant ---
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_USE_HTTPS: bool = False
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "documents_kb"
    # Points per upsert request. Large documents are chunked into batches
    # of this size so a single request never carries the whole document.
    QDRANT_UPSERT_BATCH_SIZE: int = 256

    # --- Embeddings ---
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIMENSION: int = 768  # must match the model above
    EMBEDDING_BATCH_SIZE: int = 32
    # Max unique texts held in the text -> vector LRU cache. Bounded so
    # cached embeddings can't grow unbounded and eat RAM in long-running
    # workers.
    EMBEDDING_CACHE_SIZE: int = 4096

    # --- Chunking ---
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

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


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings accessor. Using lru_cache ensures the .env file
    and environment are parsed only once per process.
    """
    return Settings()


settings = get_settings()
