"""
Reranker abstraction.

A reranker optionally re-orders the results returned by vector search so
that the top-K sent to the prompt is the best-K. The default
`IdentityReranker` keeps the retrieval order and is used when reranking is
disabled (`RERANK_ENABLED=false`); `CrossEncoderReranker` (in
cross_encoder_reranker.py) is the real scoring implementation behind the
flag.
"""
from typing import Protocol, Sequence

from src.repositories.vector_repository import SearchResult


class Reranker(Protocol):
    """Re-orders search results by relevance to the query."""

    def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Return the `top_k` most relevant results in order."""
        ...


class IdentityReranker:
    """No-op reranker: preserves retrieval order and slices to top_k."""

    def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        return list(results[:top_k])
