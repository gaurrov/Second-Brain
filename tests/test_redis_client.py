"""
Unit tests for the Redis client + circuit-breaking cache
(src/core/redis_client.py).

No Redis server is required: CacheClient is exercised against a stub whose
methods raise RedisError, verifying the circuit breaker opens, stays open,
re-probes, and that get_or_set degrades to producer().
"""
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from src.core import redis_client as redis_module
from src.core.redis_client import CacheClient, close_redis, get_redis


class _FakeClock:
    """Stand-in for time.monotonic() that can be advanced."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class _FakeRedis:
    """Records operations; raises RedisError while `failing` is True.

    `value` is what `get` returns when healthy (None == cache miss).
    """

    def __init__(self) -> None:
        self.operations: list[tuple] = []
        self.failing = False
        self.value: str | None = None

    def get(self, key: str):
        self.operations.append(("get", key))
        if self.failing:
            raise RedisConnectionError("redis is down")
        return self.value

    def set(self, key: str, value, ex=None) -> None:
        self.operations.append(("set", key))
        if self.failing:
            raise RedisConnectionError("redis is down")

    def delete(self, key: str) -> None:
        self.operations.append(("delete", key))
        if self.failing:
            raise RedisConnectionError("redis is down")


@pytest.fixture
def fake_clock(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(redis_module.time, "monotonic", clock)
    return clock


class TestCircuitBreaker:
    def test_opens_after_a_failure_and_stays_closed(self, fake_clock):
        fake = _FakeRedis()
        fake.failing = True
        cache = CacheClient(fake, prefix="t", default_ttl=60)

        assert cache.get("k") is None
        assert len(fake.operations) == 1

        # Circuit is open: redis is not consulted again inside the window.
        assert cache.get("k") is None
        assert len(fake.operations) == 1
        assert cache._available is False

    def test_reopens_after_retry_window_with_a_single_probe(self, fake_clock):
        fake = _FakeRedis()
        fake.failing = True
        fake.value = '{"cached": true}'
        cache = CacheClient(fake, prefix="t", default_ttl=60)
        cache.get("k")  # opens the circuit

        fake.failing = False
        fake_clock.advance(redis_module._CIRCUIT_RETRY_AFTER_SECONDS + 1)

        # Exactly one probe goes through and the circuit closes again.
        assert cache.get("k") == '{"cached": true}'
        assert len(fake.operations) == 2
        assert cache._available is True

    def test_failed_probe_reopens_the_circuit(self, fake_clock):
        fake = _FakeRedis()
        fake.failing = True
        cache = CacheClient(fake, prefix="t", default_ttl=60)
        cache.get("k")  # open circuit, first failure

        fake_clock.advance(redis_module._CIRCUIT_RETRY_AFTER_SECONDS + 1)
        assert cache.get("k") is None  # probe attempt fails again
        assert cache._available is False
        assert len(fake.operations) == 2


class TestGetOrSet:
    def test_falls_back_to_producer_when_redis_unavailable(self, fake_clock):
        fake = _FakeRedis()
        fake.failing = True
        cache = CacheClient(fake, prefix="t", default_ttl=60)
        producer_calls = []

        def producer():
            producer_calls.append(1)
            return {"answer": 42}

        assert cache.get_or_set("key", ttl=60, producer=producer) == {"answer": 42}
        assert producer_calls == [1]

    def test_serves_cached_value_without_calling_producer(self, fake_clock):
        fake = _FakeRedis()
        fake.value = '{"cached": true}'
        cache = CacheClient(fake, prefix="t", default_ttl=60)

        def producer():
            raise AssertionError("producer must not run on a cache hit")

        assert cache.get_or_set("key", ttl=60, producer=producer) == {"cached": True}

    def test_stores_produced_value_under_namespaced_key(self, fake_clock):
        fake = _FakeRedis()
        cache = CacheClient(fake, prefix="t", default_ttl=60)
        cache.get_or_set("key", ttl=60, producer=lambda: "fresh")
        assert ("set", "t:key") in fake.operations


class TestGetRedis:
    def test_returns_none_when_redis_disabled(self, monkeypatch):
        redis_module.get_redis.cache_clear()
        try:
            monkeypatch.setattr(redis_module.settings, "REDIS_ENABLED", False)
            assert redis_module.get_redis() is None
        finally:
            # Leave no cached value behind for later tests.
            redis_module.get_redis.cache_clear()


def test_close_redis_before_any_client_created_does_not_raise():
    get_redis.cache_clear()  # force a genuinely "never created" state
    close_redis()  # must not raise
