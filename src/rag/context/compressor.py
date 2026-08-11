"""
Context compression.

Retrieved chunks are compressed before they go into the prompt so the
prompt stays within a strict character budget (`CONTEXT_MAX_CHARACTERS`):

1. Sort by similarity score, highest first (Qdrant already returns them
   that way, but ordering is enforced here so the compressor is correct
   regardless of its input order).
2. Drop near-duplicate chunks (`CONTEXT_DEDUPE_THRESHOLD`) — overlapping
   chunks from the same document often repeat content, and the duplicates
   contribute nothing but tokens.
3. Keep chunks in score order until the budget is exhausted, truncating
   the final chunk if it would overflow the budget.

The output (`CompressedContext`) carries the full kept chunk metadata so
downstream consumers can render sources and cite provenance.
"""
import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

from src.core.config import settings
from src.repositories.vector_repository import SearchResult

logger = logging.getLogger("second_brain.context_compressor")

_TRUNCATION_SUFFIX = "…"


@dataclass(frozen=True)
class ContextChunk:
    """A single compressed chunk ready for prompt inclusion / citation."""

    document_id: str
    filename: str
    page_number: int | None
    chunk_index: int
    score: float
    content: str


@dataclass(frozen=True)
class CompressedContext:
    chunks: list[ContextChunk]
    total_characters: int
    truncated: bool


class ContextCompressor:
    def __init__(
        self,
        max_characters: int | None = None,
        dedupe_threshold: float | None = None,
    ) -> None:
        self.max_characters = max_characters or settings.CONTEXT_MAX_CHARACTERS
        self.dedupe_threshold = (
            dedupe_threshold if dedupe_threshold is not None else settings.CONTEXT_DEDUPE_THRESHOLD
        )

    def compress(self, results: Sequence[SearchResult]) -> CompressedContext:
        if not results:
            return CompressedContext(chunks=[], total_characters=0, truncated=False)

        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        deduped = self._dedupe(sorted_results)

        chunks: list[ContextChunk] = []
        used = 0
        truncated = False

        for result in deduped:
            remaining = self.max_characters - used
            if remaining <= 0:
                truncated = True
                break

            content = result.content
            if len(content) > remaining:
                # Truncate the final chunk to the remaining budget, keeping
                # a visible marker so the model knows it was cut.
                available = max(remaining - len(_TRUNCATION_SUFFIX), 0)
                if available == 0:
                    truncated = True
                    break
                content = content[:available].rstrip() + _TRUNCATION_SUFFIX
                truncated = True
                chunks.append(self._to_chunk(result, content))
                used += len(content)
                break

            chunks.append(self._to_chunk(result, content))
            used += len(content)

        return CompressedContext(chunks=chunks, total_characters=used, truncated=truncated)

    def _dedupe(self, results: Sequence[SearchResult]) -> list[SearchResult]:
        """Drop near-duplicate results, keeping the higher-scoring copy."""
        kept: list[SearchResult] = []
        for result in results:
            if any(self._similarity(result.content, other.content) >= self.dedupe_threshold for other in kept):
                continue
            kept.append(result)
        return kept

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        if a in b or b in a:
            return 1.0
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _to_chunk(result: SearchResult, content: str) -> ContextChunk:
        return ContextChunk(
            document_id=result.document_id,
            filename=result.filename,
            page_number=result.page_number,
            chunk_index=result.chunk_index,
            score=result.score,
            content=content,
        )
