"""
Rate limiting (production-ready).

Implements a fixed-window rate limit using Redis atomic INCR + EXPIRE
(``RATE_LIMIT_STRATEGY=fixed_window``), with an in-process fallback store
so the limiter still functions when Redis is not configured or briefly
unavailable.

Fail-open vs fail-closed: when Redis is unreachable the limiter degrades
to the in-memory store rather than rejecting traffic, keeping availability
the priority for a read-heavy RAG backend. The ``login``/``refresh``
endpoints use a stricter limit (brute-force protection), everything else
uses the default.
"""
import logging
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from src.core.config import settings
from src.core.metrics import record_rejection

logger = logging.getLogger("second_brain.rate_limit")

_WARN_THROTTLE_SECONDS = 30.0


class RateLimitStore(Protocol):
    def increment(self, key: str, ttl_seconds: int) -> int:
        """Atomically increment `key` and return the new value, expiring it after `ttl_seconds`."""
        ...


class RedisRateLimitStore:
    def __init__(self, redis) -> None:
        self._redis = redis
        self._fallback = MemoryRateLimitStore()
        self._last_warn = 0.0

    def increment(self, key: str, ttl_seconds: int) -> int:
        try:
            count = self._redis.incr(key)
            if count == 1:
                self._redis.expire(key, ttl_seconds)
            return int(count)
        except Exception as exc:  # noqa: BLE001 - degrade to memory store
            now = time.monotonic()
            if now - self._last_warn >= _WARN_THROTTLE_SECONDS:
                self._last_warn = now
                logger.warning("Redis unavailable for rate limiting, using in-memory fallback: %s", exc)
            return self._fallback.increment(key, ttl_seconds)


class MemoryRateLimitStore:
    """Thread-safe in-process store for environments without Redis."""

    def __init__(self) -> None:
        self._counts: dict[tuple[str, int], tuple[int, float]] = {}
        self._lock = threading.Lock()

    def increment(self, key: str, ttl_seconds: int) -> int:
        window = int(time.monotonic() // ttl_seconds)
        with self._lock:
            entry = self._counts.get((key, window))
            now = time.monotonic()
            if entry is None:
                self._counts[(key, window)] = (1, now)
                return 1
            count, _created = entry
            new_count = count + 1
            self._counts[(key, window)] = (new_count, _created)
            # Opportunistically evict stale windows.
            if len(self._counts) > 10_000:
                for k in [k for k, (_, c) in self._counts.items() if now - c > ttl_seconds * 2]:
                    self._counts.pop(k, None)
            return new_count


@dataclass(frozen=True)
class RateLimitPolicy:
    """How many requests a client may make per window before rejection."""

    limit: int
    window_seconds: int


class FixedWindowRateLimiter:
    def __init__(
        self,
        store: RateLimitStore,
        *,
        default_policy: RateLimitPolicy,
        login_policy: RateLimitPolicy,
        key_prefix: str = "ratelimit",
    ) -> None:
        self._store = store
        self._default_policy = default_policy
        self._login_policy = login_policy
        self._key_prefix = key_prefix

    def check(self, client_id: str, *, login: bool = False) -> bool:
        """
        Record one attempt for `client_id` and return True if the client
        is still within its limit for the current window.
        """
        policy = self._login_policy if login else self._default_policy
        window = int(time.time() // policy.window_seconds)
        tag = "login" if login else "default"
        key = f"{self._key_prefix}:{client_id}:{tag}:{window}"
        count = self._store.increment(key, policy.window_seconds)
        allowed = count <= policy.limit
        if not allowed:
            record_rejection("rate_limit")
        return allowed


def build_rate_limiter() -> FixedWindowRateLimiter:
    """Construct the configured limiter, preferring Redis when enabled."""
    from src.core.redis_client import get_redis

    redis = get_redis()
    store: RateLimitStore
    if redis is not None:
        store = RedisRateLimitStore(redis)
    else:
        store = MemoryRateLimitStore()

    return FixedWindowRateLimiter(
        store,
        default_policy=RateLimitPolicy(
            limit=settings.RATE_LIMIT_DEFAULT_LIMIT,
            window_seconds=settings.RATE_LIMIT_DEFAULT_WINDOW_SECONDS,
        ),
        login_policy=RateLimitPolicy(
            limit=settings.RATE_LIMIT_LOGIN_LIMIT,
            window_seconds=settings.RATE_LIMIT_LOGIN_WINDOW_SECONDS,
        ),
        key_prefix=f"{settings.REDIS_CACHE_PREFIX}:ratelimit",
    )


@lru_cache(maxsize=1)
def get_rate_limiter() -> FixedWindowRateLimiter:
    """Process-wide rate limiter singleton (built once per process)."""
    return build_rate_limiter()
