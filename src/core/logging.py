"""
Structured logging.

Production logging is JSON-lines so it can be shipped straight into
CloudWatch/ELK/Loki/Promtail and correlated via the ``request_id`` field
injected from ``src.core.context``.

Usage stays identical to stdlib logging:

    logger = logging.getLogger("second_brain.something")
    logger.info("job done", extra={"document_id": doc.id, "elapsed_ms": 12.3})

Every record gets: ``ts``, ``level``, ``logger``, ``message``, plus any
`extra` keys and ``request_id``. Exceptions render as ``exc_info``.

``setup_logging()`` is idempotent and must be called exactly once, as
early as possible in the process (first thing in ``src.main``).
"""
import json
import logging
import logging.config
import sys
import time
from typing import Any

from src.core.context import get_request_id
from src.core.config import settings

_LOGGER_NAME = "second_brain"
_SETUP_KEY = "_second_brain_logging_configured"


class JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "ts_ms": round(record.created * 1000),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = get_request_id()
        if request_id:
            payload["request_id"] = request_id

        # Merge any structured `extra={...}` keys. Skip fields the logging
        # machinery reserves for itself.
        reserved = {
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "module", "msecs",
            "message", "msg", "name", "pathname", "process", "processName",
            "relativeCreated", "stack_info", "taskName", "thread", "threadName",
        }
        for key, value in record.__dict__.items():
            if key not in reserved and not key.startswith("_"):
                payload[key] = _json_safe(value)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable fallback for local development (LOG_FORMAT=text)."""

    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record)} | {record.levelname} | {record.name} | {record.getMessage()}"
        request_id = get_request_id()
        if request_id:
            base = f"[req={request_id}] {base}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of a log value to something JSON-serializable."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def setup_logging() -> None:
    """Install the structured logging configuration for the whole process."""
    root = logging.getLogger()
    if getattr(root, _SETUP_KEY, False):
        return  # already configured (e.g. reloaded dev servers)

    formatter: logging.Formatter
    if settings.LOG_FORMAT == "json":
        formatter = JsonFormatter()
    else:
        formatter = TextFormatter()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level_int)

    # Silence third-party chatter unless the app is in DEBUG mode.
    if not settings.DEBUG:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("qdrant_client").setLevel(logging.WARNING)

    setattr(root, _SETUP_KEY, True)


__all__ = [
    "JsonFormatter",
    "TextFormatter",
    "setup_logging",
]
