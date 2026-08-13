"""
Shared fixtures / options for the benchmark suite.

Benchmarks live outside ``tests/`` (testpaths = ["tests"] in pyproject.toml),
so the normal ``pytest`` run never collects them. Run them with:

    pytest benchmarks/ --benchmark-only            # no live infra required
    pytest benchmarks/ --benchmark-only --run-live # also embed with real model weights

Anything that needs real model weights (embedding throughput, end-to-end
ingestion) is marked ``live`` and reuses the exact gating pattern from
``tests/conftest.py``: without ``--run-live`` those benchmarks are skipped.
"""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run live benchmarks (require real model weights).",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-live"):
        skip_live = pytest.mark.skip(
            reason="requires --run-live (real model weights)"
        )
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)
