"""
Unit tests for ContextCompressor.

Verifies score ordering, near-duplicate removal, the character budget
(including truncation of the overflowing final chunk), and empty input.
"""
from src.rag.context.compressor import ContextCompressor
from src.repositories.vector_repository import SearchResult


def _result(
    content: str,
    score: float,
    *,
    document_id: str = "doc-1",
    filename: str = "notes.txt",
    page: int | None = 1,
    chunk_index: int = 0,
) -> SearchResult:
    return SearchResult(
        point_id=f"point-{chunk_index}",
        score=score,
        user_id="user-1",
        document_id=document_id,
        filename=filename,
        page_number=page,
        chunk_index=chunk_index,
        content=content,
        timestamp="2026-08-11T00:00:00+00:00",
    )


class TestOrdering:
    def test_sorts_by_score_descending_regardless_of_input_order(self):
        compressor = ContextCompressor(max_characters=100_000, dedupe_threshold=1.0)
        results = [
            _result("low relevance", 0.3, chunk_index=0),
            _result("high relevance", 0.95, chunk_index=1),
            _result("medium relevance", 0.6, chunk_index=2),
        ]
        compressed = compressor.compress(results)
        assert [c.score for c in compressed.chunks] == [0.95, 0.6, 0.3]
        assert compressed.chunks[0].content == "high relevance"

    def test_empty_input(self):
        compressed = ContextCompressor().compress([])
        assert compressed.chunks == []
        assert compressed.total_characters == 0
        assert compressed.truncated is False


class TestBudget:
    def test_keeps_all_chunks_within_budget(self):
        compressor = ContextCompressor(max_characters=100, dedupe_threshold=1.0)
        results = [_result("x" * 40, 0.9, chunk_index=0), _result("y" * 40, 0.8, chunk_index=1)]
        compressed = compressor.compress(results)
        assert len(compressed.chunks) == 2
        assert compressed.total_characters == 80
        assert compressed.total_characters <= 100
        assert compressed.truncated is False

    def test_drops_lowest_scoring_chunks_when_over_budget(self):
        compressor = ContextCompressor(max_characters=60, dedupe_threshold=1.0)
        results = [_result("a" * 40, 0.9, chunk_index=0), _result("b" * 40, 0.8, chunk_index=1)]
        compressed = compressor.compress(results)
        # The first chunk (40 chars) fits; the second is truncated to the
        # 20 remaining characters instead of being dropped wholesale.
        assert len(compressed.chunks) == 2
        assert compressed.chunks[0].content == "a" * 40
        assert compressed.chunks[1].content == "b" * 19 + "…"
        assert compressed.truncated is True
        assert compressed.total_characters <= 60

    def test_truncates_overflowing_chunk_to_budget(self):
        compressor = ContextCompressor(max_characters=50, dedupe_threshold=1.0)
        results = [_result("z" * 100, 0.9, chunk_index=0)]
        compressed = compressor.compress(results)
        assert len(compressed.chunks) == 1
        assert compressed.truncated is True
        assert len(compressed.chunks[0].content) <= 50
        assert compressed.chunks[0].content.endswith("…")


class TestDedupe:
    def test_near_duplicate_chunks_dropped(self):
        compressor = ContextCompressor(max_characters=100_000, dedupe_threshold=0.9)
        results = [
            _result("The deployment runbook explains how to roll back.", 0.95, chunk_index=0),
            _result("The deployment runbook explains how to roll back.", 0.90, chunk_index=1),
            _result("Completely different unrelated content.", 0.85, chunk_index=2),
        ]
        compressed = compressor.compress(results)
        assert len(compressed.chunks) == 2
        assert compressed.chunks[0].score == 0.95
        assert compressed.chunks[1].content == "Completely different unrelated content."

    def test_substring_duplicates_are_dropped(self):
        compressor = ContextCompressor(max_characters=100_000, dedupe_threshold=0.5)
        results = [
            _result("The quick brown fox jumps over the lazy dog.", 0.9, chunk_index=0),
            _result("The quick brown fox jumps", 0.7, chunk_index=1),
        ]
        compressed = compressor.compress(results)
        assert len(compressed.chunks) == 1
        assert compressed.chunks[0].score == 0.9
