"""
Unit tests for retry/backoff (src/utils/retry.py).

The `backoff` callable is injected everywhere so no test ever sleeps. The
explicit (attempt, base_delay, max_delay, jitter) argument contract is
verified by recording the calls.
"""
import asyncio

import pytest

from src.utils import retry as retry_module
from src.utils.retry import async_retry, retry

_NOOP_BACKOFF = lambda *args, **kwargs: 0.0  # noqa: E731


def _recorded_backoff(seen: list):
    def backoff(attempt: int, base_delay: float, max_delay: float, jitter: bool) -> float:
        seen.append((attempt, base_delay, max_delay, jitter))
        return 0.0

    return backoff


class TestSyncRetry:
    def test_succeeds_on_first_try_without_sleeping(self):
        calls = []
        backoff_calls = []

        @retry(max_attempts=3, backoff=_recorded_backoff(backoff_calls))
        def ok():
            calls.append(1)
            return "done"

        assert ok() == "done"
        assert len(calls) == 1
        assert backoff_calls == []

    def test_retries_retryable_exception_up_to_max_attempts(self):
        calls = []

        @retry(max_attempts=3, backoff=_NOOP_BACKOFF)
        def flaky():
            calls.append(1)
            raise ConnectionError("transient failure")

        with pytest.raises(ConnectionError, match="transient failure"):
            flaky()
        assert len(calls) == 3

    def test_does_not_retry_non_retryable_exception(self):
        calls = []

        @retry(max_attempts=3, backoff=_NOOP_BACKOFF)
        def bad_input():
            calls.append(1)
            raise ValueError("client error, never retried")

        with pytest.raises(ValueError, match="client error, never retried"):
            bad_input()
        assert len(calls) == 1

    def test_custom_retry_on_predicate_governs_what_is_retried(self):
        calls = []

        @retry(
            max_attempts=3,
            backoff=_NOOP_BACKOFF,
            retry_on=lambda exc: isinstance(exc, TimeoutError),
        )
        def flaky():
            calls.append(1)
            raise TimeoutError("retry me")

        with pytest.raises(TimeoutError):
            flaky()
        assert len(calls) == 3

    def test_backoff_receives_expected_arguments(self):
        seen = []

        @retry(
            max_attempts=3,
            base_delay=0.5,
            max_delay=5.0,
            jitter=False,
            backoff=_recorded_backoff(seen),
        )
        def flaky():
            raise ConnectionError("down")

        with pytest.raises(ConnectionError):
            flaky()

        # Two retries before the final attempt => backoff called for
        # attempts 1 and 2 with the resolved base/cap/jitter values.
        assert seen == [(1, 0.5, 5.0, False), (2, 0.5, 5.0, False)]


class TestAsyncRetry:
    def test_async_succeeds_on_first_try_without_sleeping(self):
        calls = []
        backoff_calls = []

        @async_retry(max_attempts=3, backoff=_recorded_backoff(backoff_calls))
        async def ok():
            calls.append(1)
            return "done"

        assert asyncio.run(ok()) == "done"
        assert len(calls) == 1
        assert backoff_calls == []

    def test_async_retries_up_to_max_attempts_then_raises(self):
        calls = []

        @async_retry(max_attempts=3, backoff=_NOOP_BACKOFF)
        async def flaky():
            calls.append(1)
            raise TimeoutError("slow dependency")

        with pytest.raises(TimeoutError, match="slow dependency"):
            asyncio.run(flaky())
        assert len(calls) == 3

    def test_async_does_not_retry_non_retryable_exception(self):
        calls = []

        @async_retry(max_attempts=3, backoff=_NOOP_BACKOFF)
        async def bad():
            calls.append(1)
            raise ValueError("bad request")

        with pytest.raises(ValueError, match="bad request"):
            asyncio.run(bad())
        assert len(calls) == 1

    def test_async_backoff_receives_expected_arguments(self):
        seen = []

        @async_retry(
            max_attempts=3,
            base_delay=0.2,
            max_delay=2.0,
            jitter=True,
            backoff=_recorded_backoff(seen),
        )
        async def flaky():
            raise ConnectionError("down")

        with pytest.raises(ConnectionError):
            asyncio.run(flaky())

        assert seen == [(1, 0.2, 2.0, True), (2, 0.2, 2.0, True)]


class TestBackoff:
    def test_default_backoff_caps_exponential_growth(self):
        assert retry_module._default_backoff(1, 0.2, 5.0, jitter=False) == pytest.approx(0.2)
        # Exponential growth is capped at max_delay.
        assert retry_module._default_backoff(10, 0.2, 5.0, jitter=False) == pytest.approx(5.0)

    def test_default_backoff_with_jitter_stays_in_range(self):
        value = retry_module._default_backoff(2, 0.2, 5.0, jitter=True)
        assert 0.0 <= value <= 0.4  # base * 2^(2-1)
