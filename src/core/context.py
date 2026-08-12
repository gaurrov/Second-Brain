"""
Per-request context.

Uses a ``contextvars.ContextVar`` so the current request ID is available
anywhere in the request's async/sync call graph without being threaded
through every function signature. Set once by the request middleware in
``src/main.py``; read by the structured log formatter and by anything that
needs to correlate with the current request (metrics, health checks, ...).

NOTE: contextvars do NOT flow into arbitrary worker threads. Background
tasks that want to carry the request ID must copy the context
(``contextvars.copy_context()``) at submission time or log their own
correlation key (e.g. document_id).
"""
from contextvars import ContextVar, Token

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_id(request_id: str | None) -> Token:
    """Bind `request_id` to the current execution context. Returns a token."""
    return _request_id.set(request_id)


def reset_request_id(token: Token) -> None:
    """Restore the previous request ID using the token from set_request_id."""
    _request_id.reset(token)


def get_request_id() -> str | None:
    """The request ID of the current execution context, if any."""
    return _request_id.get()
