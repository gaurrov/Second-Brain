from src.core.config import settings
from src.rag.rerankers.base import IdentityReranker, Reranker
from src.rag.rerankers.cross_encoder_reranker import CrossEncoderReranker


def build_reranker() -> Reranker:
    """Return the reranker configured by `RERANK_ENABLED`."""
    if settings.RERANK_ENABLED:
        return CrossEncoderReranker()
    return IdentityReranker()


__all__ = [
    "CrossEncoderReranker",
    "IdentityReranker",
    "Reranker",
    "build_reranker",
]
