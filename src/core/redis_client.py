"""
Redis connectivity + a resilient cache client.

Two layers:

1. ``get_redis()`` — a process-wide ``redis.Redis`` instance built from
   ``REDIS_URL`` with a bounded connection pool and tight socket timeouts.
   Returns ``None`` when ``REDIS_ENABLED=false`` so the rest of the app
   degrades to "no Redis" without special-casing.

2. ``CacheClient`` — a thin, JSON-aware cache with a circuit breaker:
   Redis failures are logged (throttled), never raised, and the cache
   temporarily stops being consulted so a down Redis cannot add a socket
   timeout to every request. It re-probes after a short backoff and
   recovers automatically.

The cache is a fast-path enhancement, never a source of truth — callers
must be able to operate with a cache miss / a dead Redis.
"""
import json
import logging
import time
from functools import lru_cache
from typing import Any, Callable, TypeVar

from redis import Redis
from redis.exceptions import RedisError

from src.core.config import settings
from src.core.metrics import (
    redis_cache_hits_total,
    redis_cache_misses_total,
    redis_errors_total,
    redis_operation_duration_seconds,
)

logger = logging.getLogger("second_brain.redis")

T = TypeVar("T")

# After a failure the cache stays open (not consulted) for this long before
# allowing a probe request through.
_CIRCUIT_RETRY_AFTER_SECONDS = 10.0
# Log at most one warning per this many seconds while Redis is down.
_WARN_THROTTLE_SECONDS = 30.0


@lru_cache(maxsize=1)
def get_redis() -> Redis | None:
    """Process-wide Redis client, or None when REDIS_ENABLED=false."""
    if not settings.REDIS_ENABLED:
        return None
    logger.info(
        "Connecting to Redis at %s (timeout=%.1fs)",
        settings.REDIS_URL,
        settings.REDIS_SOCKET_TIMEOUT_SECONDS,
    )
    return Redis.from_url(
        settings.REDIS_URL,
        socket_timeout=settings.REDIS_SOCKET_TIMEOUT_SECONDS,
        socket_connect_timeout=settings.REDIS_SOCKET_TIMEOUT_SECONDS,
        socket_keepalive=True,
        max_connections=max(settings.REDIS_CONNECTION_POOL_SIZE, 1),
        decode_responses=True,
    )


def get_cache() -> "CacheClient | None":
    """Return the shared CacheClient, or None when Redis is disabled."""
    redis = get_redis()
    if redis is None:
        return None
    return CacheClient(
        redis,
        prefix=settings.REDIS_CACHE_PREFIX,
        default_ttl=settings.REDIS_DEFAULT_TTL_SECONDS,
    )


def close_redis() -> None:
    """Close the process-wide Redis connection pool (shutdown hook)."""
    if get_redis.cache_info().currsize == 0:
        return
    try:
        get_redis().close()
    except Exception:  # noqa: BLE001 - best effort on shutdown
        pass
    get_redis.cache_clear()


def _namespace(prefix: str, key: str) -> str:
    return f"{prefix}:{key}"


class CacheClient:
    """JSON-aware Redis cache with a circuit breaker and metrics."""

    def __init__(
        self,
        redis: Redis,
        *,
        prefix: str,
        default_ttl: int,
    ) -> None:
        self._redis = redis
        self._prefix = prefix
        self._default_ttl = default_ttl
        self._available = True
        self._last_failure = 0.0
        self._last_warn = 0.0

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------
    def _circuit_open(self) -> bool:
        if self._available:
            return False
        if time.monotonic() - self._last_failure >= _CIRCUIT_RETRY_AFTER_SECONDS:
            self._available = True  # allow one probe through
            return False
        return True

    def _record_failure(self, operation: str) -> None:
        self._available = False
        self._last_failure = time.monotonic()
        redis_errors_total.labels(operation=operation).inc()
        now = time.monotonic()
        if now - self._last_warn >= _WARN_THROTTLE_SECONDS:
            self._last_warn = now
            logger.warning("Redis %s failed; cache disabled for %.0fs", operation, _CIRCUIT_RETRY_AFTER_SECONDS)

    def _run(self, operation: str, fn: Callable[[], T]) -> T | None:
        """Execute a Redis op through the circuit breaker; never raises."""
        if self._circuit_open():
            return None
        try:
            with redis_operation_duration_seconds.labels(operation=operation).time():
                result = fn()
            self._available = True
            return result
        except RedisError as exc:  # noqa: BLE001 - cache must never raise
            self._record_failure(operation)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ping(self) -> bool:
        """Probe connectivity regardless of circuit state."""
        try:
            with redis_operation_duration_seconds.labels(operation="ping").time():
                return bool(self._redis.ping())
        except RedisError:
            self._record_failure("ping")
            return False

    def get(self, key: str) -> str | None:
        value = self._run("get", lambda: self._redis.get(self._full_key(key)))
        if value is not None:
            redis_cache_hits_total.inc()
        else:
            redis_cache_misses_total.inc()
        return value

    def set(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        self._run("set", lambda: self._redis.set(self._full_key(key), value, ex=ttl or self._default_ttl))

    def delete(self, key: str) -> None:
        self._run("delete", lambda: self._redis.delete(self._full_key(key)))

    def get_json(self, key: str) -> Any | None:
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            redis_cache_misses_total.inc()
            return None

    def set_json(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        self.set(key, json.dumps(value), ttl=ttl)

    def get_or_set(self, key: str, ttl: int, producer: Callable[[], T]) -> T:
        """
        Return the cached value for `key` or compute it via `producer`,
        store it, and return it. Falls back to `producer()` whenever Redis
        is unavailable or the value cannot be serialized.
        """
        cached = self.get_json(key)
        if cached is not None:
            return cached
        value = producer()
        try:
            self.set_json(key, value, ttl=ttl)
        except Exception:  # noqa: BLE001 - producer result is already computed
            logger.debug("Could not store cache entry for %s", key)
        return value

    def close(self) -> None:
        try:
            self._redis.close()
        except Exception:  # noqa: BLE001 - best effort
            pass

    def _full_key(self, key: str) -> str:
        return _namespace(self._prefix, key)
