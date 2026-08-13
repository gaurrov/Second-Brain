"""
Unit tests for the background worker pool (src/workers/pool.py).

Pure in-process tests: submits real tasks to real worker threads and uses
threading.Event for deterministic completion signals. No external infra.
"""
import threading
import time

from src.workers.pool import WorkerPool

_TIMEOUT = 10.0


def _wait(event: threading.Event) -> None:
    assert event.wait(timeout=_TIMEOUT), "task did not complete in time"


class TestWorkerPool:
    def test_submit_executes_task_on_a_worker(self):
        pool = WorkerPool(concurrency=2, maxsize=10, shutdown_timeout=5)
        pool.start()
        try:
            results = []
            done = threading.Event()

            def task():
                results.append("ran")
                done.set()

            assert pool.submit(task) is True
            _wait(done)
            assert results == ["ran"]
            assert pool.running is True
        finally:
            pool.shutdown(wait=True)

    def test_shutdown_drains_queued_tasks_before_returning(self):
        pool = WorkerPool(concurrency=2, maxsize=100, shutdown_timeout=10)
        pool.start()
        results = []
        lock = threading.Lock()

        def task(index: int):
            time.sleep(0.01)
            with lock:
                results.append(index)

        submitted = [pool.submit(task, index) for index in range(20)]
        assert all(submitted)

        pool.shutdown(wait=True)
        assert sorted(results) == list(range(20))

    def test_failing_task_does_not_kill_the_worker_thread(self):
        pool = WorkerPool(concurrency=1, maxsize=10, shutdown_timeout=5)
        pool.start()
        try:
            second_done = threading.Event()

            def boom():
                raise RuntimeError("background task exploded")

            def survivor():
                second_done.set()

            assert pool.submit(boom) is True
            assert pool.submit(survivor) is True

            # The failing task runs first (FIFO); the survivor must still run
            # on the same worker thread afterwards.
            _wait(second_done)
        finally:
            pool.shutdown(wait=True)

    def test_shutdown_is_idempotent(self):
        pool = WorkerPool(concurrency=1, maxsize=10, shutdown_timeout=5)
        pool.start()
        pool.shutdown(wait=True)
        pool.shutdown(wait=True)  # second call must not raise

    def test_submit_on_non_running_pool_executes_inline_and_returns_false(self):
        pool = WorkerPool(concurrency=2, maxsize=10, shutdown_timeout=5)
        results = []

        returned = pool.submit(lambda: results.append("inline"))

        assert returned is False
        assert results == ["inline"]
        assert pool.running is False
