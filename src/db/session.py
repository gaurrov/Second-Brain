"""
SQLAlchemy engine and session factory.

`get_db` is a FastAPI dependency generator: it yields a session and
guarantees it is closed after the request, regardless of success or
failure. Endpoints/services never construct sessions manually — they
always receive one via dependency injection.
"""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import settings

_engine_kwargs: dict[str, object] = {
    "pool_pre_ping": True,
    "future": True,
}
if not settings.sqlalchemy_database_uri.startswith("sqlite"):
    _engine_kwargs["pool_size"] = 10
    _engine_kwargs["max_overflow"] = 20

engine = create_engine(
    settings.sqlalchemy_database_uri,
    **_engine_kwargs,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    future=True,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a DB session and closes it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
