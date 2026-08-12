"""
Timing helpers.

Small context managers used to measure hot paths (embedding, Qdrant calls,
pipeline steps) and optionally record the elapsed time to a metric or a
callback — without try/finally boilerplate everywhere.
"""
import time
from collections.abc import Callable
from typing import Any


class Timer:
    """Measures wall-clock elapsed time of a with-block."""

    __slots__ = ("_start", "_stop")

    def __init__(self) -> None:
        self._start: float = 0.0
        self._stop: float | None = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Seconds between enter and exit. Raises if not yet exited."""
        if self._stop is None:
            raise RuntimeError("Timer not stopped yet.")
        return self._stop - self._start


def time_it(on_done: Callable[[float], Any] | None = None) -> Timer:
    """
    Return a Timer that invokes ``on_done(elapsed_seconds)`` when the block
    exits. Use for recording metrics without touching the metric objects:
        with time_it(lambda s: metric.observe(s)):
            do_work()
    """
    timer = Timer()
    _original_exit = timer.__exit__

    def _patched_exit(*args: Any) -> None:
        _original_exit(*args)
        if on_done is not None:
            on_done(timer.elapsed)

    timer.__exit__ = _patched_exit  # type: ignore[method-assign]
    return timer
