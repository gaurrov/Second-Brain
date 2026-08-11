"""
Cross-encoder reranker (optional).

Uses a sentence-transformers `CrossEncoder` to score each (query, chunk)
pair directly, which is substantially more accurate than vector-only
similarity for judging "does this chunk actually answer the question".

Gated behind `RERANK_ENABLED` (default false) because loading a second
model costs RAM. The model is imported lazily and cached per-instance so
this module can be imported (and unit-tested) without any model weights.
"""
import logging
from functools import lru_cache
from typing import Sequence

from src.core.config import settings
from src.repositories.vector_repository import SearchResult

logger = logging.getLogger("second_brain.reranker")


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.RERANK_MODEL_NAME

    def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        if not results:
            return []
        model = self._get_model()
        pairs = [(query, result.content) for result in results]
        scores = model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(scores, results), key=lambda pair: pair[0], reverse=True)
        return [result for _, result in ranked[:top_k]]

    @property
    def model(self):
        """Lazily loaded (and cached) CrossEncoder instance."""
        return self._get_model()

    @lru_cache(maxsize=1)
    def _get_model(self):
        from sentence_transformers import CrossEncoder  # lazy: no weights unless used

        logger.info("Loading cross-encoder reranker %s", self.model_name)
        return CrossEncoder(self.model_name)
