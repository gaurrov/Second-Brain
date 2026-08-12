"""
Health checks.

- ``/health/live``  — is the process alive (no external calls).
- ``/health/ready`` — can the process serve traffic (DB, Qdrant, and —
                       when enabled — Redis are all reachable).

Readiness probes each dependency with a cheap call and returns a
per-component breakdown so operators can see *which* dependency is down,
not just a binary 200/503. Individual checks are isolated: one failing
dependency can never crash the probe.
"""
import logging
from dataclasses import dataclass

from sqlalchemy import text

from src.core.config import settings

logger = logging.getLogger("second_brain.health")


@dataclass(frozen=True)
class ComponentHealth:
    name: str
    ok: bool
    detail: str | None = None
    latency_ms: float = 0.0


class HealthChecker:
    def __init__(self) -> None:
        self._required_components = ("database", "qdrant")
        self._optional_components = ("redis",)

    def check_live(self) -> dict:
        return {"status": "ok", "app": settings.APP_NAME, "env": settings.APP_ENV}

    def check_ready(self) -> tuple[dict, bool]:
        """
        Run every dependency probe. Returns (report, ready) where `ready`
        is False when any *required* component is unhealthy.
        """
        checks = [
            self._check_database(),
            self._check_qdrant(),
            self._check_redis(),
        ]

        report: dict = {
            "status": "ok",
            "app": settings.APP_NAME,
            "env": settings.APP_ENV,
            "components": {
                check.name: {"ok": check.ok, "latency_ms": check.latency_ms}
                if check.detail is None
                else {"ok": check.ok, "detail": check.detail, "latency_ms": check.latency_ms}
                for check in checks
            },
        }
        ready = all(check.ok for check in checks if check.name in self._required_components)
        if not ready:
            report["status"] = "degraded"
        return report, ready

    def _check_database(self) -> ComponentHealth:
        from src.db.session import engine

        start = _now_ms()
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return ComponentHealth("database", True, latency_ms=_elapsed(start))
        except Exception as exc:  # noqa: BLE001 - probe must not raise
            logger.warning("Database readiness probe failed: %s", exc)
            return ComponentHealth("database", False, detail=str(exc), latency_ms=_elapsed(start))

    def _check_qdrant(self) -> ComponentHealth:
        from src.vectorstore.qdrant_client import get_qdrant_client

        start = _now_ms()
        try:
            client = get_qdrant_client()
            client.get_collections()
            return ComponentHealth("qdrant", True, latency_ms=_elapsed(start))
        except Exception as exc:  # noqa: BLE001 - probe must not raise
            logger.warning("Qdrant readiness probe failed: %s", exc)
            return ComponentHealth("qdrant", False, detail=str(exc), latency_ms=_elapsed(start))

    def _check_redis(self) -> ComponentHealth:
        if not settings.REDIS_ENABLED:
            return ComponentHealth("redis", True, detail="disabled")
        from src.core.redis_client import get_redis

        start = _now_ms()
        redis = get_redis()
        if redis is None:
            return ComponentHealth("redis", False, detail="not configured", latency_ms=_elapsed(start))
        try:
            redis.ping()
            return ComponentHealth("redis", True, latency_ms=_elapsed(start))
        except Exception as exc:  # noqa: BLE001 - probe must not raise
            return ComponentHealth("redis", False, detail=str(exc), latency_ms=_elapsed(start))


def _now_ms() -> float:
    import time

    return time.perf_counter() * 1000


def _elapsed(start_ms: float) -> float:
    return round(_now_ms() - start_ms, 2)


_health_checker: HealthChecker | None = None


def get_health_checker() -> HealthChecker:
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker
