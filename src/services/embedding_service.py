"""
Embedding generation service.

Wraps a Sentence-Transformers model (BGE by default). Production design
goals:

  - SINGLETON MODEL LOADING: model weights are loaded exactly once per
    process (module-level ``lru_cache``). Loading a transformer weights
    file is expensive and must never happen per request or per chunk.
  - CACHING: an LRU cache keyed by normalized text memoizes text ->
    vector results. This skips inference entirely for repeated content
    (overlapping chunks, repeated user queries) and is bounded so it
    can't grow without limit in long-running workers.
  - BATCHED + BOUNDED MEMORY: inputs are deduplicated before inference,
    and cache-misses are encoded in fixed-size sub-batches so peak memory
    stays proportional to one batch rather than to the whole document.
  - THREAD-SAFETY: model.encode() is serialized behind a lock so
    concurrent ingestion tasks never race on shared model state.

BGE-en v1.5 models use an instruction-prefix convention: passages are
embedded as-is, but retrieval queries are prefixed with a query
instruction so the query embedding lands in the same region of space it
was trained to. ``embed_documents`` / ``embed_query`` keep these two paths
separate so ingestion and retrieval can't accidentally use the wrong one.
"""
import logging
import threading
from collections import OrderedDict
from functools import lru_cache
from typing import Sequence

from sentence_transformers import SentenceTransformer

from src.core.config import settings

logger = logging.getLogger("second_brain.embedding")

Vector = list[float]

# Instruction prefix used by BAAI bge-en v1.5 models for retrieval queries.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Process-wide singleton SentenceTransformer, lazily loaded once."""
    logger.info("Loading embedding model %s", settings.EMBEDDING_MODEL_NAME)
    return SentenceTransformer(settings.EMBEDDING_MODEL_NAME)


def clear_model_cache() -> None:
    """Drop the cached model so the next call reloads it (used in tests)."""
    _get_model.cache_clear()


class _LRUCache:
    """Small thread-safe LRU cache used for text -> vector memoization."""

    __slots__ = ("_data", "_maxsize", "_lock")

    def __init__(self, maxsize: int) -> None:
        self._data: OrderedDict[str, Vector] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: str) -> Vector | None:
        with self._lock:
            value = self._data.get(key)
            if value is None:
                return None
            self._data.move_to_end(key)
            # Return a copy so callers mutating a result can't corrupt the
            # shared cache entry.
            return list(value)

    def set(self, key: str, value: Vector) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


class EmbeddingService:
    QUERY_INSTRUCTION = QUERY_INSTRUCTION

    def __init__(
        self,
        model: SentenceTransformer | None = None,
        *,
        cache_size: int | None = None,
    ) -> None:
        """
        Args:
            model: Optional pre-constructed model. When omitted, the
                process-wide singleton is used (and lazily loaded on
                first embed). Injecting a model is how tests avoid
                downloading real weights.
            cache_size: Optional override for the LRU cache size. Defaults
                to settings.EMBEDDING_CACHE_SIZE.
        """
        self._model = model
        self._encode_lock = threading.Lock()
        self._cache = _LRUCache(max(cache_size or settings.EMBEDDING_CACHE_SIZE, 1))

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = _get_model()
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding dimension of the underlying model."""
        return int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """
        Embed a batch of passage/chunk texts for storage.

        Identical texts are embedded once and replayed from the cache;
        only cache-misses are passed through the model, in sub-batches of
        ``EMBEDDING_BATCH_SIZE`` to keep peak memory bounded.
        """
        if not texts:
            return []

        normalized = [text.strip() for text in texts]
        results: list[Vector | None] = [None] * len(normalized)
        miss_positions_by_text: dict[str, list[int]] = {}

        for position, text in enumerate(normalized):
            cached = self._cache.get(text)
            if cached is not None:
                results[position] = cached
            else:
                # Same-position dedupe: a text repeated inside this call
                # is collected once and replayed for every occurrence.
                miss_positions_by_text.setdefault(text, []).append(position)

        if miss_positions_by_text:
            distinct_misses = list(miss_positions_by_text)
            vectors = self._encode_batches(distinct_misses)
            for text, vector in zip(distinct_misses, vectors):
                self._cache.set(text, vector)
                for position in miss_positions_by_text[text]:
                    results[position] = list(vector)

        return [result for result in results if result is not None]

    def embed_query(self, text: str) -> Vector:
        """
        Embed a single user query for similarity search.

        Applies the BGE query instruction prefix so retrieval queries are
        embedded the same way they were during model training.
        """
        query_text = f"{self.QUERY_INSTRUCTION}{text.strip()}"

        cached = self._cache.get(query_text)
        if cached is not None:
            return cached

        with self._encode_lock:
            encoded = self.model.encode(
                [query_text],
                batch_size=1,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        vector = encoded[0].tolist()
        self._cache.set(query_text, vector)
        return vector

    def clear_cache(self) -> None:
        """Drop all cached embeddings (memory release + test isolation)."""
        self._cache.clear()

    def cache_size(self) -> int:
        """Number of texts currently memoized (for tests / observability)."""
        return len(self._cache)

    def _encode_batches(self, texts: list[str]) -> list[Vector]:
        """
        Encode `texts` in sub-batches of EMBEDDING_BATCH_SIZE, returning
        the full vector list. Processing sub-batches (rather than handing
        the whole list to the model in one call) bounds the peak output
        tensor size to one batch, which matters for very large documents.
        """
        batch_size = max(settings.EMBEDDING_BATCH_SIZE, 1)
        vectors: list[Vector] = []
        for start in range(0, len(texts), batch_size):
            sub_batch = texts[start : start + batch_size]
            with self._encode_lock:
                encoded = self.model.encode(
                    sub_batch,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            vectors.extend(encoded.tolist())
        return vectors
