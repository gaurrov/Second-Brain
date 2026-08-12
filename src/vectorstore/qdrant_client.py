"""
Qdrant client factory.

Exposes a single, process-wide QdrantClient instance (constructed lazily
on first use) so connection setup isn't repeated per-request. Nothing
outside `src/vectorstore/` and `src/repositories/vector_repository.py`
should import qdrant_client directly — everyone else goes through the
vector repository.

Retries against transient transport failures happen in the vector
repository (per operation), not here. This module only owns client
construction (timeouts, gRPC preference) and shutdown.
"""
import logging
from functools import lru_cache

from qdrant_client import QdrantClient

from src.core.config import settings

logger = logging.getLogger("second_brain.vectorstore")


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Return a process-wide QdrantClient singleton (lazily constructed)."""
    logger.info(
        "Connecting to Qdrant at %s:%s (https=%s, grpc=%s)",
        settings.QDRANT_HOST,
        settings.QDRANT_PORT,
        settings.QDRANT_USE_HTTPS,
        settings.QDRANT_GRPC_ENABLED,
    )
    return QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        https=settings.QDRANT_USE_HTTPS,
        api_key=settings.QDRANT_API_KEY,
        timeout=settings.QDRANT_TIMEOUT_SECONDS,
        grpc_port=settings.QDRANT_GRPC_PORT,
        prefer_grpc=settings.QDRANT_GRPC_ENABLED,
    )


def close_qdrant_client() -> None:
    """Close the process-wide Qdrant client and drop it (graceful shutdown)."""
    if get_qdrant_client.cache_info().currsize == 0:
        return
    client = get_qdrant_client()
    if client is None:
        return
    try:
        client.close()
    except Exception:  # noqa: BLE001 - best effort on shutdown
        logger.debug("Qdrant client close failed (ignored)", exc_info=True)
    get_qdrant_client.cache_clear()
