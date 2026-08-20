"""
RQ (Redis Queue) integration.

Provides:
  - ``rq_queue()``  -- process-wide RQ Queue singleton
  - ``dispatch_to_rq()`` -- callable that enqueues via RQ instead of running inline
  - ``run_rq_worker()`` -- entry point for a standalone RQ worker process

Design:
  The FastAPI app calls ``dispatch_to_rq(document_id)`` which pushes the
  job onto Redis.  A separate worker process picks it up, calls
  ``process_document_task(document_id)``, and stores the result.
"""
import logging
import signal
import sys
from functools import lru_cache

from rq import Queue, Worker

from src.core.config import settings

logger = logging.getLogger("second_brain.workers.rq")


@lru_cache(maxsize=1)
def rq_queue() -> Queue:
    """Process-wide RQ Queue singleton backed by the app's Redis."""
    from src.core.redis_client import get_redis

    redis = get_redis()
    if redis is None:
        raise RuntimeError(
            "Redis is required for TASK_WORKER=rq but REDIS_ENABLED=false."
        )
    return Queue(
        name=settings.RQ_QUEUE_NAME,
        is_async=True,
        connection=redis,
    )


def dispatch_to_rq(document_id) -> None:
    """
    Enqueue ``process_document_task(document_id)`` onto the RQ queue.
    """
    from src.services.ingestion_service import process_document_task

    queue = rq_queue()
    job = queue.enqueue(
        process_document_task,
        document_id,
        job_timeout=settings.RQ_DEFAULT_TIMEOUT,
        result_ttl=settings.RQ_RESULT_TTL,
    )
    logger.info(
        "Enqueued document %s as RQ job %s (queue=%s)",
        document_id, job.id, settings.RQ_QUEUE_NAME,
    )


def run_rq_worker() -> None:
    """
    Entry point for the RQ worker process.
    """
    from src.core.redis_client import get_redis

    redis = get_redis()
    if redis is None:
        logger.error("Redis is required for RQ worker but REDIS_ENABLED=false")
        sys.exit(1)

    queue = Queue(name=settings.RQ_QUEUE_NAME, connection=redis)
    worker = Worker(
        queues=[queue],
        connection=redis,
        name=f"secondbrain-worker-{settings.RQ_QUEUE_NAME}",
    )

    def _shutdown(signum, frame):
        logger.info("Received signal %s, shutting down worker...", signum)
        worker.handle_force_stop()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info(
        "Starting RQ worker on queue '%s' (timeout=%ds)",
        settings.RQ_QUEUE_NAME,
        settings.RQ_DEFAULT_TIMEOUT,
    )
    worker.work()


__all__ = ["rq_queue", "dispatch_to_rq", "run_rq_worker"]
