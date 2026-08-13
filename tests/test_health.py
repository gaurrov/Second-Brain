"""
Unit tests for health checks (src/core/health.py).

All probes are exercised with stubs/monkeypatching — no DB, Qdrant or Redis
server is contacted. Verifies liveness never touches external systems, the
readiness matrix (required vs. optional components), and that a raising probe
is contained (never propagates out of check_ready).
"""
import pytest

from src.core import health as health_module
from src.core.health import HealthChecker

import src.db.session as db_session_module
import src.vectorstore.qdrant_client as qdrant_module
import src.core.redis_client as redis_module


class _OkConnection:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, statement):
        return None


class _OkEngine:
    def connect(self):
        return _OkConnection()


class _FailingEngine:
    def connect(self):
        raise RuntimeError("database is down")


class _OkQdrantClient:
    def get_collections(self):
        return None


class _FailingQdrantClient:
    def get_collections(self):
        raise ConnectionError("qdrant is down")


class _FailingRedisClient:
    def ping(self):
        raise ConnectionError("redis is down")


def _patch_externals(monkeypatch, *, engine, qdrant, redis):
    """Point every external probe at the given stubs."""
    monkeypatch.setattr(db_session_module, "engine", engine)
    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: qdrant)
    monkeypatch.setattr(redis_module, "get_redis", lambda: redis)


class TestLiveness:
    def test_check_live_never_touches_external_systems(self, monkeypatch):
        # Every external dependency is wired to raise: any accidental probe
        # inside check_live would surface as an error here.
        _patch_externals(
            monkeypatch,
            engine=_FailingEngine(),
            qdrant=_FailingQdrantClient(),
            redis=_FailingRedisClient(),
        )
        monkeypatch.setattr(health_module.settings, "REDIS_ENABLED", True)

        report = HealthChecker().check_live()
        assert report["status"] == "ok"
        assert report["app"]
        assert report["env"]


class TestReadiness:
    def test_all_healthy_reports_ready(self, monkeypatch):
        monkeypatch.setattr(health_module.settings, "REDIS_ENABLED", False)
        _patch_externals(
            monkeypatch,
            engine=_OkEngine(),
            qdrant=_OkQdrantClient(),
            redis=lambda: pytest.fail("redis must not be consulted when disabled"),
        )

        report, ready = HealthChecker().check_ready()
        assert ready is True
        assert report["status"] == "ok"
        for component in ("database", "qdrant", "redis"):
            assert report["components"][component]["ok"] is True

    def test_database_failure_makes_ready_false_but_redis_fine(self, monkeypatch):
        monkeypatch.setattr(health_module.settings, "REDIS_ENABLED", False)
        _patch_externals(
            monkeypatch,
            engine=_FailingEngine(),
            qdrant=_OkQdrantClient(),
            redis=lambda: pytest.fail("redis must not be consulted when disabled"),
        )

        report, ready = HealthChecker().check_ready()
        assert ready is False
        assert report["status"] == "degraded"
        assert report["components"]["database"]["ok"] is False
        assert report["components"]["qdrant"]["ok"] is True
        assert report["components"]["redis"]["ok"] is True

    def test_ready_true_when_redis_disabled_even_if_redis_would_fail(self, monkeypatch):
        monkeypatch.setattr(health_module.settings, "REDIS_ENABLED", False)
        _patch_externals(
            monkeypatch,
            engine=_OkEngine(),
            qdrant=_OkQdrantClient(),
            redis=lambda: pytest.fail("redis must not be consulted when disabled"),
        )

        report, ready = HealthChecker().check_ready()
        assert ready is True
        assert report["components"]["redis"]["detail"] == "disabled"

    def test_raising_probe_never_propagates(self, monkeypatch):
        monkeypatch.setattr(health_module.settings, "REDIS_ENABLED", True)
        _patch_externals(
            monkeypatch,
            engine=_FailingEngine(),
            qdrant=_FailingQdrantClient(),
            redis=_FailingRedisClient(),
        )

        report, ready = HealthChecker().check_ready()  # must not raise
        assert ready is False
        for component in ("database", "qdrant", "redis"):
            assert report["components"][component]["ok"] is False
            assert report["components"][component]["detail"]
