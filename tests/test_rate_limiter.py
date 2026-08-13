"""
Unit tests for the rate limiter (src/core/rate_limiter.py).

Covers the in-memory store (window rollover, eviction at 10k+ entries), the
fixed-window limiter (allows up to the limit, rejects the N+1th request in a
window) and the Redis store's failover to the in-memory store when Redis
errors. No Redis server is required — the Redis store is exercised with a
stub whose methods raise.
"""
import pytest

from src.core import rate_limiter as rate_limiter_module


class _FakeClock:
    """Stand-in for time.monotonic() / time.time() that can be advanced."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


@pytest.fixture
def fake_clock(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(rate_limiter_module.time, "monotonic", clock)
    monkeypatch.setattr(rate_limiter_module.time, "time", clock)
    return clock


class TestMemoryRateLimitStore:
    def test_window_rollover_resets_count(self, fake_clock):
        store = rate_limiter_module.MemoryRateLimitStore()
        assert store.increment("client-a", ttl_seconds=60) == 1
        assert store.increment("client-a", ttl_seconds=60) == 2

        fake_clock.advance(61)  # next 60s window

        assert store.increment("client-a", ttl_seconds=60) == 1
        assert store.increment("client-a", ttl_seconds=60) == 2

    def test_distinct_clients_have_independent_counts(self, fake_clock):
        store = rate_limiter_module.MemoryRateLimitStore()
        store.increment("alice", ttl_seconds=60)
        store.increment("alice", ttl_seconds=60)
        assert store.increment("bob", ttl_seconds=60) == 1

    def test_evicts_stale_entries_when_over_10k(self, fake_clock):
        """Crossing the 10_000-entry threshold prunes stale windows."""
        store = rate_limiter_module.MemoryRateLimitStore()
        ttl = 60

        # Backdate 10_000 distinct (key, window) entries so they count as
        # stale (older than ttl*2) when the eviction pass runs, and add one
        # live entry that we then re-increment.
        for index in range(10_000):
            store._counts[(f"stale-{index}", 0)] = (1, fake_clock() - ttl * 3)
        store._counts[("hitting", 0)] = (1, fake_clock())

        # Incrementing an existing key crosses 10_000 entries, which runs
        # the opportunistic eviction of every stale entry.
        assert store.increment("hitting", ttl_seconds=ttl) == 2
        assert len(store._counts) == 1
        assert list(store._counts) == [("hitting", 0)]


class TestFixedWindowRateLimiter:
    def _limiter(self, limit: int = 3) -> rate_limiter_module.FixedWindowRateLimiter:
        return rate_limiter_module.FixedWindowRateLimiter(
            rate_limiter_module.MemoryRateLimitStore(),
            default_policy=rate_limiter_module.RateLimitPolicy(limit=limit, window_seconds=60),
            login_policy=rate_limiter_module.RateLimitPolicy(limit=10, window_seconds=60),
            key_prefix="test",
        )

    def test_allows_up_to_limit_then_rejects_nth_plus_one(self):
        limiter = self._limiter(limit=3)
        assert [limiter.check("client-1") for _ in range(3)] == [True, True, True]
        assert limiter.check("client-1") is False

    def test_other_clients_unaffected_by_a_limited_client(self):
        limiter = self._limiter(limit=2)
        limiter.check("client-1")
        limiter.check("client-1")
        assert limiter.check("client-1") is False
        assert limiter.check("client-2") is True

    def test_window_rollover_resets_the_budget(self, fake_clock):
        limiter = self._limiter(limit=3)
        for _ in range(3):
            assert limiter.check("client-1") is True
        assert limiter.check("client-1") is False

        fake_clock.advance(61)
        assert limiter.check("client-1") is True


class TestRedisRateLimitStore:
    def test_falls_back_to_memory_store_on_redis_error(self):
        class _FailingRedis:
            def __init__(self):
                self.incr_calls = 0

            def incr(self, key):
                self.incr_calls += 1
                raise ConnectionError("redis unreachable")

        failing = _FailingRedis()
        store = rate_limiter_module.RedisRateLimitStore(failing)

        assert store.increment("client-a", ttl_seconds=60) == 1
        assert store.increment("client-a", ttl_seconds=60) == 2
        assert failing.incr_calls == 2
        # The fallback store is the one doing the counting.
        assert len(store._fallback._counts) == 1

    def test_recovers_to_redis_when_it_comes_back(self):
        class _FlakyRedis:
            def __init__(self):
                self.up = False

            def incr(self, key):
                if not self.up:
                    raise ConnectionError("still down")
                return 7

        redis = _FlakyRedis()
        store = rate_limiter_module.RedisRateLimitStore(redis)

        assert store.increment("k", 60) == 1  # falls back
        redis.up = True
        assert store.increment("k", 60) == 7  # redis is authoritative again
