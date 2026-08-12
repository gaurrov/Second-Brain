"""
Prometheus metrics.

Exposes the standard ``/metrics`` endpoint data (see ``src/main.py``) plus
a small helper API used across the codebase to record HTTP, DB, Qdrant,
Redis, LLM, embedding and ingestion timing/counters.

Path labels are normalized (UUIDs -> ``{id}``) before being used as labels
so metric cardinality stays bounded under real traffic.

All metric objects are created at import time but only populated when
``METRICS_ENABLED=true``; ``observe_duration`` returns a no-op context
manager when disabled so call sites don't need to branch.
"""
import logging
import re
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.core.config import settings

logger = logging.getLogger("second_brain.metrics")

_ENABLED = settings.METRICS_ENABLED

_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_LONG_INT_RE = re.compile(r"/\d{6,}")


def normalize_path(path: str) -> str:
    """Replace volatile path segments (UUIDs, long ids) with stable labels."""
    path = _UUID_RE.sub("{id}", path)
    path = _LONG_INT_RE.sub("/{id}", path)
    return path or "/"


# --------------------------------------------------------------------------
# HTTP / middleware
# --------------------------------------------------------------------------
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests handled.",
    ["method", "path", "status"],
)
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency.",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
http_requests_in_flight = Gauge(
    "http_requests_in_flight",
    "HTTP requests currently being processed.",
    ["method", "path"],
)
http_rejected_total = Counter(
    "http_rejected_total",
    "Requests rejected by a policy (e.g. rate limiting).",
    ["reason"],
)


def record_request(method: str, path: str, status_code: int, duration_seconds: float) -> None:
    if not _ENABLED:
        return
    label_path = normalize_path(path)
    http_requests_total.labels(method=method, path=label_path, status=status_code).inc()
    http_request_duration_seconds.labels(method=method, path=label_path).observe(duration_seconds)


def record_rejection(reason: str) -> None:
    if _ENABLED:
        http_rejected_total.labels(reason=reason).inc()


# --------------------------------------------------------------------------
# Generic timing helper
# --------------------------------------------------------------------------
def observe_duration(metric: Histogram, *label_values: Any) -> Iterator[None]:
    """
    Context manager that times its body and records the duration on
    `metric`. No-op (zero overhead) when metrics are disabled.
    """
    if not _ENABLED:
        return nullcontext()
    return _Timed(metric, label_values)


class _Timed:
    __slots__ = ("_metric", "_labels", "_start")

    def __init__(self, metric: Histogram, label_values: tuple) -> None:
        self._metric = metric
        self._labels = label_values
        self._start: float | None = None

    def __enter__(self) -> "_Timed":
        import time

        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        import time

        if self._start is not None:
            self._metric.labels(*self._labels).observe(time.perf_counter() - self._start)


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
db_query_duration_seconds = Histogram(
    "db_query_duration_seconds",
    "SQLAlchemy statement execution time.",
    ["operation"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
db_slow_queries_total = Counter(
    "db_slow_queries_total",
    "SQL statements slower than DB_SLOW_QUERY_THRESHOLD_MS.",
    ["operation"],
)

# --------------------------------------------------------------------------
# Vector store (Qdrant)
# --------------------------------------------------------------------------
vector_operations_total = Counter(
    "vector_operations_total",
    "Qdrant operations attempted.",
    ["operation", "outcome"],
)
vector_operation_duration_seconds = Histogram(
    "vector_operation_duration_seconds",
    "Qdrant operation latency.",
    ["operation"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
vector_points_upserted_total = Counter(
    "vector_points_upserted_total",
    "Qdrant points written.",
    ["operation"],
)

# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------
embedding_requests_total = Counter(
    "embedding_requests_total",
    "Embedding generation requests.",
    ["kind"],
)
embedding_encode_duration_seconds = Histogram(
    "embedding_encode_duration_seconds",
    "Model encode() latency per batch.",
    ["kind"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
embedding_cache_hits_total = Counter(
    "embedding_cache_hits_total",
    "Text -> vector cache hits (L1 memory + L2 redis).",
    ["tier"],
)
embedding_cache_misses_total = Counter(
    "embedding_cache_misses_total",
    "Text -> vector cache misses.",
    ["tier"],
)

# --------------------------------------------------------------------------
# Redis
# --------------------------------------------------------------------------
redis_operation_duration_seconds = Histogram(
    "redis_operation_duration_seconds",
    "Redis command latency.",
    ["operation"],
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1),
)
redis_errors_total = Counter(
    "redis_errors_total",
    "Redis command failures (cache degrades to a no-op).",
    ["operation"],
)
redis_cache_hits_total = Counter(
    "redis_cache_hits_total",
    "Redis cache get() hits.",
)
redis_cache_misses_total = Counter(
    "redis_cache_misses_total",
    "Redis cache get() misses / errors.",
)

# --------------------------------------------------------------------------
# LLM (Groq)
# --------------------------------------------------------------------------
llm_requests_total = Counter(
    "llm_requests_total",
    "LLM completion requests.",
    ["outcome"],
)
llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "LLM completion latency.",
    buckets=(0.1, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120),
)

# --------------------------------------------------------------------------
# Outbound retries
# --------------------------------------------------------------------------
outbound_retries_total = Counter(
    "outbound_retries_total",
    "Transient failures retried per external target.",
    ["target", "outcome"],
)

# --------------------------------------------------------------------------
# Ingestion pipeline
# --------------------------------------------------------------------------
ingestion_documents_total = Counter(
    "ingestion_documents_total",
    "Documents processed by the ingestion pipeline.",
    ["status"],
)
ingestion_pipeline_duration_seconds = Histogram(
    "ingestion_pipeline_duration_seconds",
    "Ingestion pipeline step latency.",
    ["step"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)
ingestion_pipeline_total_seconds = Histogram(
    "ingestion_pipeline_total_seconds",
    "Total ingestion duration per document.",
    buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
)

# --------------------------------------------------------------------------
# Background worker pool
# --------------------------------------------------------------------------
task_queue_depth = Gauge(
    "task_queue_depth",
    "Number of tasks waiting in the background worker queue.",
)
task_processed_total = Counter(
    "task_processed_total",
    "Background tasks executed.",
    ["outcome"],
)


def metrics_payload() -> tuple[str, bytes]:
    """Return (content_type, body) for the Prometheus scrape endpoint."""
    return CONTENT_TYPE_LATEST, generate_latest()
