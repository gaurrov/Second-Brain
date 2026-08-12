"""
Background worker pool.

A minimal, dependency-free task queue that decouples "dispatch work" from
"how the work runs". It powers ``TASK_WORKER=pool``: a small set of daemon
threads drain a bounded queue, so a slow ingestion job can never stall a
request and is no longer coupled to FastAPI's BackgroundTasks (which are
per-request and tied to the response lifecycle).

This is intentionally shaped like the seam that a real broker-based queue
(Celery/RQ) would occupy: `dispatch_processing` in ``src/api/deps.py`` is
the only call site that changes when a broker arrives.

Lifecycle: ``start()`` at app startup, ``shutdown(wait=True)`` on graceful
shutdown. Tasks are drained on shutdown so in-flight ingestion completes
(or is cut off by ``WORKER_SHUTDOWN_TIMEOUT_SECONDS``).
"""
import logging
import queue
import threading
import time
from functools import lru_cache
from typing import Any, Callable

from src.core.config import settings
from src.core.metrics import task_processed_total, task_queue_depth

logger = logging.getLogger("second_brain.workers")

_SENTINEL = object()


class WorkerPool:
    """Fixed-size thread pool draining a bounded task queue."""

    def __init__(
        self,
        concurrency: int,
        maxsize: int,
        shutdown_timeout: int,
    ) -> None:
        self._concurrency = max(concurrency, 1)
        self._maxsize = max(maxsize, 1)
        self._shutdown_timeout = shutdown_timeout
        self._queue: queue.Queue = queue.Queue(maxsize=self._maxsize)
        self._threads: list[threading.Thread] = []
        self._running = False
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def queued(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        """Spawn the worker threads. Idempotent."""
        with self._lock:
            if self._running:
                return
            self._running = True
            for index in range(self._concurrency):
                thread = threading.Thread(
                    target=self._worker,
                    name=f"sb-worker-{index}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)
            logger.info(
                "Worker pool started: %d workers, queue size %d",
                self._concurrency, self._maxsize,
            )

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
        """
        Enqueue `fn(*args, **kwargs)`. Returns True when enqueued. If the
        pool is not running the task runs inline (best-effort, keeps the
        operation correct during startup edge cases); if the queue is
        full the task is dropped and logged.
        """
        if not self._running:
            logger.warning("Worker pool not running; executing task inline")
            self._execute(fn, args, kwargs)
            return False
        try:
            self._queue.put_nowait((fn, args, kwargs))
            task_queue_depth.inc()
            return True
        except queue.Full:
            task_processed_total.labels(outcome="rejected").inc()
            logger.error("Worker queue full (%d); dropping task", self._maxsize)
            return False

    def shutdown(self, *, wait: bool = True) -> None:
        """
        Stop accepting new work and (optionally) drain in-flight tasks.
        Idempotent.
        """
        with self._lock:
            if not self._running:
                return
            self._running = False

        for _ in self._threads:
            self._queue.put(_SENTINEL)

        deadline = time.monotonic() + self._shutdown_timeout
        if wait:
            while self._queue.unfinished_tasks > 0 and time.monotonic() < deadline:
                time.sleep(0.05)

        for thread in self._threads:
            thread.join(timeout=max(0.1, self._shutdown_timeout))
        self._threads.clear()
        logger.info("Worker pool shut down")

    # ------------------------------------------------------------------
    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            task_queue_depth.dec()
            if item is _SENTINEL:
                self._queue.task_done()
                break
            fn, args, kwargs = item
            try:
                self._execute(fn, args, kwargs)
            finally:
                self._queue.task_done()

    @staticmethod
    def _execute(fn: Callable[..., Any], args: tuple, kwargs: dict) -> None:
        try:
            fn(*args, **kwargs)
            task_processed_total.labels(outcome="success").inc()
        except Exception:  # noqa: BLE001 - worker must never die
            task_processed_total.labels(outcome="error").inc()
            logger.exception("Background task %s failed", getattr(fn, "__name__", fn))


@lru_cache(maxsize=1)
def get_worker_pool() -> WorkerPool:
    """Process-wide worker pool singleton (started by the app lifespan)."""
    return WorkerPool(
        concurrency=settings.WORKER_CONCURRENCY,
        maxsize=settings.WORKER_QUEUE_MAXSIZE,
        shutdown_timeout=settings.WORKER_SHUTDOWN_TIMEOUT_SECONDS,
    )


def stop_worker_pool() -> None:
    """Graceful shutdown hook for the process-wide pool."""
    pool = get_worker_pool.cache_info().currsize and get_worker_pool()
    if pool is None:
        return
    pool.shutdown(wait=True)
    get_worker_pool.cache_clear()


__all__ = ["WorkerPool", "get_worker_pool", "stop_worker_pool"]
