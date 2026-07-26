"""
Qdrant client factory.

Exposes a single, process-wide QdrantClient instance (constructed lazily
on first use) so connection setup isn't repeated per-request. Nothing
outside `src/vectorstore/` and `src/repositories/vector_repository.py`
should import qdrant_client directly — everyone else goes through the
vector repository.
"""
from functools import lru_cache

from qdrant_client import QdrantClient

from src.core.config import settings


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Return a process-wide QdrantClient singleton (lazily constructed)."""
    return QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        https=settings.QDRANT_USE_HTTPS,
        api_key=settings.QDRANT_API_KEY,
    )
