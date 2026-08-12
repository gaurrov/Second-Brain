"""Background worker abstractions (see src.workers.pool)."""
from src.workers.pool import WorkerPool, get_worker_pool, stop_worker_pool

__all__ = ["WorkerPool", "get_worker_pool", "stop_worker_pool"]
