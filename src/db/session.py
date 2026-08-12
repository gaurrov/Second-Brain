"""
SQLAlchemy engine and session factory.

`get_db` is a FastAPI dependency generator: it yields a session and
guarantees it is closed after the request, regardless of success or
failure. Endpoints/services never construct sessions manually — they
always receive one via dependency injection.

Production settings:

- Connection pooling tuned by ``DB_POOL_SIZE`` / ``DB_MAX_OVERFLOW`` /
  ``DB_POOL_TIMEOUT_SECONDS`` / ``DB_POOL_RECYCLE_SECONDS``.
- ``pool_pre_ping`` so stale connections are discarded instead of serving
  broken requests after a Postgres restart.
- Statement-level timing: every query is timed, Prometheus metrics are
  recorded, and queries slower than ``DB_SLOW_QUERY_THRESHOLD_MS`` are
  logged so missing indexes / N+1s are visible in production.
- ``dispose_engine()`` is the graceful-shutdown hook (drains + closes the
  pool after the event loop stops accepting requests).
"""
import logging
import time
from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import settings
from src.core.metrics import db_query_duration_seconds, db_slow_queries_total

logger = logging.getLogger("second_brain.db")

_slow_threshold_seconds = settings.DB_SLOW_QUERY_THRESHOLD_MS / 1000.0


def _build_engine_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {
        "pool_pre_ping": True,
        "future": True,
        "echo": settings.DB_ECHO,
    }
    if not settings.sqlalchemy_database_uri.startswith("sqlite"):
        kwargs["pool_size"] = max(settings.DB_POOL_SIZE, 1)
        kwargs["max_overflow"] = max(settings.DB_MAX_OVERFLOW, 0)
        kwargs["pool_timeout"] = max(settings.DB_POOL_TIMEOUT_SECONDS, 1)
        kwargs["pool_recycle"] = max(settings.DB_POOL_RECYCLE_SECONDS, 60)
    return kwargs


engine = create_engine(
    settings.sqlalchemy_database_uri,
    **_build_engine_kwargs(),
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    future=True,
)


def _operation_label(statement: str) -> str:
    """Coarse operation category (SELECT/INSERT/UPDATE/DELETE/other) for metrics."""
    first = statement.lstrip().split(None, 1)[0].upper() if statement.strip() else "OTHER"
    if first in {"SELECT", "INSERT", "UPDATE", "DELETE"}:
        return first
    return "OTHER"


@event.listens_for(engine, "before_cursor_execute")
def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany) -> None:
    conn.info.setdefault("_sb_query_start", []).append(time.perf_counter())
    conn.info.setdefault("_sb_query_stmt", []).append(statement)


@event.listens_for(engine, "after_cursor_execute")
def _after_cursor_execute(conn, cursor, statement, parameters, context, executemany) -> None:
    starts = conn.info.get("_sb_query_start", [])
    stmts = conn.info.get("_sb_query_stmt", [])
    if not starts:
        return
    start = starts.pop(0)
    stmt = stmts.pop(0)
    elapsed = time.perf_counter() - start

    operation = _operation_label(stmt)
    db_query_duration_seconds.labels(operation=operation).observe(elapsed)
    if elapsed >= _slow_threshold_seconds:
        db_slow_queries_total.labels(operation=operation).inc()
        snippet = " ".join(stmt.split())[:200]
        logger.warning(
            "Slow query %.1fms (%s): %s...",
            elapsed * 1000,
            operation,
            snippet,
        )


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a DB session and closes it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def dispose_engine() -> None:
    """Close all pooled DB connections (graceful shutdown)."""
    try:
        engine.dispose()
    except Exception:  # noqa: BLE001 - best effort on shutdown
        logger.debug("DB engine disposal failed (ignored)", exc_info=True)
