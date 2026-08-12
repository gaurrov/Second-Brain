"""
Retry with exponential backoff and full jitter.

Used for transient, idempotent operations against external systems
(Qdrant, Redis, Groq). Failures that are retryable (timeouts, connection
errors, 5xx) are retried up to ``RETRY_MAX_ATTEMPTS`` with an exponential
backoff capped at ``RETRY_MAX_DELAY_SECONDS`` and randomized with full
jitter so concurrent workers don't stampede the dependency in lockstep.

Design notes:

- Exceptions from the *final* attempt propagate unchanged — the caller
  still owns error handling (e.g. translating to a domain exception).
- ``retry_on`` lets callers whitelist retryable failures; everything else
  fails immediately (e.g. a 400 is never retried).
- A ``backoff`` callable can be injected so tests run without sleeping.
"""
import asyncio
import functools
import logging
import random
import time
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from src.core.config import settings
from src.core.metrics import outbound_retries_total

logger = logging.getLogger("second_brain.retry")

P = ParamSpec("P")
T = TypeVar("T")

DEFAULT_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    ConnectionResetError,
    ConnectionRefusedError,
    BrokenPipeError,
    OSError,
)


def _default_retry_on(exc: BaseException) -> bool:
    return isinstance(exc, DEFAULT_RETRYABLE_EXCEPTIONS)


def _default_backoff(attempt: int, base_delay: float, max_delay: float, jitter: bool) -> float:
    """Exponential backoff with full jitter: delay in [0, min(max, base*2^attempt))."""
    cap = min(max_delay, base_delay * (2 ** max(attempt - 1, 0)))
    if not jitter:
        return cap
    return random.uniform(0, cap)


def retry(
    fn: Callable[P, T] | None = None,
    *,
    max_attempts: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    jitter: bool | None = None,
    retry_on: Callable[[BaseException], bool] | None = None,
    backoff: Callable[..., float] | None = None,
    target: str = "unknown",
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> Callable[P, T]:
    """
    Decorator adding retry logic to a synchronous callable.

    Args:
        fn: The function to wrap.
        max_attempts: Total attempts (including the first). Defaults to
            ``RETRY_MAX_ATTEMPTS``.
        base_delay / max_delay / jitter: Backoff parameters. Defaults to
            their ``RETRY_*`` config counterparts.
        retry_on: Predicate deciding whether an exception is retryable.
            Defaults to connection/timeout/OS errors.
        backoff: ``backoff(attempt, base_delay, max_delay, jitter)`` returning
            the sleep seconds for `attempt` (1-indexed). Injected by tests.
        target: Label for retry metrics / logs (e.g. "qdrant", "groq").
        on_retry: Optional ``(attempt, exc)`` callback fired before sleeping.
    """
    if fn is None:
        return lambda f: retry(
            f,
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            jitter=jitter,
            retry_on=retry_on,
            backoff=backoff,
            target=target,
            on_retry=on_retry,
        )

    attempts = max_attempts or max(settings.RETRY_MAX_ATTEMPTS, 1)
    base = settings.RETRY_BACKOFF_BASE_SECONDS if base_delay is None else base_delay
    cap = settings.RETRY_MAX_DELAY_SECONDS if max_delay is None else max_delay
    use_jitter = settings.RETRY_JITTER if jitter is None else jitter
    should_retry = retry_on or _default_retry_on
    sleep_fn = backoff or _default_backoff

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        last_exc: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - retried per retry_on
                last_exc = exc
                if not should_retry(exc) or attempt >= attempts:
                    break
                delay = sleep_fn(attempt, base, cap, use_jitter)
                if on_retry is not None:
                    on_retry(attempt, exc)
                outbound_retries_total.labels(target=target, outcome="retried").inc()
                logger.debug(
                    "%s attempt %d/%d failed (%s); retrying in %.2fs",
                    target, attempt, attempts, exc, delay,
                )
                time.sleep(delay)

        # last_exc is never None here (an exception escaped the loop).
        assert last_exc is not None
        outbound_retries_total.labels(target=target, outcome="failed").inc()
        logger.warning(
            "%s failed after %d attempts: %s", target, attempts, last_exc
        )
        raise last_exc

    return wrapper


def async_retry(
    fn: Callable[P, Any] | None = None,
    *,
    max_attempts: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    jitter: bool | None = None,
    retry_on: Callable[[BaseException], bool] | None = None,
    backoff: Callable[..., float] | None = None,
    target: str = "unknown",
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> Callable[P, Any]:
    """Asyncio variant of :func:`retry` using ``asyncio.sleep``."""
    if fn is None:
        return lambda f: async_retry(
            f,
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            jitter=jitter,
            retry_on=retry_on,
            backoff=backoff,
            target=target,
            on_retry=on_retry,
        )

    attempts = max_attempts or max(settings.RETRY_MAX_ATTEMPTS, 1)
    base = settings.RETRY_BACKOFF_BASE_SECONDS if base_delay is None else base_delay
    cap = settings.RETRY_MAX_DELAY_SECONDS if max_delay is None else max_delay
    use_jitter = settings.RETRY_JITTER if jitter is None else jitter
    should_retry = retry_on or _default_retry_on
    sleep_fn = backoff or _default_backoff

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
        last_exc: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not should_retry(exc) or attempt >= attempts:
                    break
                delay = sleep_fn(attempt, base, cap, use_jitter)
                if on_retry is not None:
                    on_retry(attempt, exc)
                outbound_retries_total.labels(target=target, outcome="retried").inc()
                logger.debug(
                    "%s attempt %d/%d failed (%s); retrying in %.2fs",
                    target, attempt, attempts, exc, delay,
                )
                await asyncio.sleep(delay)

        assert last_exc is not None
        outbound_retries_total.labels(target=target, outcome="failed").inc()
        logger.warning(
            "%s failed after %d attempts: %s", target, attempts, last_exc
        )
        raise last_exc

    return wrapper
