# syntax=docker/dockerfile:1
#
# Multi-stage build:
#   - builder:  compiles anything that needs a toolchain (psycopg2 build
#               deps, wheel builds) into a throwaway venv.
#   - runtime:  gets ONLY the installed packages + app code. No gcc, no
#               libpq-dev, no build tools in the final image (smaller,
#               fewer CVEs to track).
#
# Build context is the project root:  docker build -f docker/Dockerfile .

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt .
# requirements-dev.txt (test tooling) is intentionally NOT installed here — see requirements-dev.txt for local/CI use.
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# psycopg2-binary bundles its own libpq, but libpq5 is kept as a safety net
# for any future package that links against the system copy. No compilers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

ENV PATH="/opt/venv/bin:${PATH}"
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY . .

# Run as an unprivileged user (never root in production).
USER appuser

EXPOSE 8000

# Liveness probe: curl may not exist in python:slim, so use urllib.
# urlopen returns 0 on success and exits non-zero (with traceback on stderr)
# when the service is unreachable — exactly what Docker needs to fail the check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"]

# NOTE on --workers: each uvicorn worker is its own OS process, and the
# background WorkerPool (src/workers/pool.py) is a per-process singleton.
# With --workers 2 every process gets its own independent pool, which is the
# right isolation (a crashed worker takes down only its own queue), at the
# cost of the two pools not sharing a single queue. If a shared, durable queue
# is ever needed, that is the moment to swap in a real broker (Celery/RQ) —
# src/api/deps.py is already shaped for that seam.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
