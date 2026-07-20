"""
Embedding generation service.

Wraps a Sentence-Transformers model (BGE by default). The model is
loaded once per process (module-level singleton via lru_cache) since
loading model weights is expensive — it must not happen per-request.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.core.config import settings


@lru_cache
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.EMBEDDING_MODEL_NAME)


class EmbeddingService:
    """
    BGE models are trained with an instruction prefix convention:
    documents/passages are embedded as-is, but queries should be prefixed
    with "Represent this sentence for searching relevant passages: " to
    match the model's training setup and get good retrieval quality. This
    service exposes separate methods so ingestion and query-time embedding
    can't accidentally use the wrong one.
    """

    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self):
        self._model = _get_model()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of passage/chunk texts for storage."""
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query for similarity search."""
        vector = self._model.encode(
            self.QUERY_INSTRUCTION + text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()
