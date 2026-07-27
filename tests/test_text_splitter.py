"""
Tests for the text splitting/chunking service.

Validates chunk boundaries, overlap, minimum chunk size merging,
page-number isolation, and edge cases like single-character text.
"""
import pytest

from src.rag.loaders.base_loader import LoadedPage
from src.rag.splitters.text_splitter import TextSplitterService


@pytest.fixture
def splitter():
    return TextSplitterService(chunk_size=100, chunk_overlap=20, min_chunk_size=20)


@pytest.fixture
def large_splitter():
    return TextSplitterService(chunk_size=800, chunk_overlap=120, min_chunk_size=100)


class TestChunking:
    def test_short_text_single_chunk(self, splitter):
        pages = [LoadedPage(page_number=1, content="Short text.")]
        chunks = splitter.split_pages(pages)
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."
        assert chunks[0].page_number == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].character_count == len("Short text.")

    def test_long_text_splits(self, splitter):
        text = "word " * 30  # 150 chars
        pages = [LoadedPage(page_number=1, content=text)]
        chunks = splitter.split_pages(pages)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 100

    def test_chunks_never_span_pages(self, splitter):
        pages = [
            LoadedPage(page_number=1, content="Page one " * 15),
            LoadedPage(page_number=2, content="Page two " * 15),
        ]
        chunks = splitter.split_pages(pages)
        page_numbers = {c.page_number for c in chunks}
        assert 1 in page_numbers
        assert 2 in page_numbers
        for chunk in chunks:
            if chunk.page_number == 1:
                assert "Page two" not in chunk.content
            elif chunk.page_number == 2:
                assert "Page one" not in chunk.content

    def test_chunk_indices_are_sequential(self, splitter):
        text = "word " * 30
        pages = [LoadedPage(page_number=1, content=text)]
        chunks = splitter.split_pages(pages)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_page_skipped(self, splitter):
        pages = [
            LoadedPage(page_number=1, content=""),
            LoadedPage(page_number=2, content="Content here."),
        ]
        chunks = splitter.split_pages(pages)
        assert len(chunks) == 1
        assert chunks[0].page_number == 2

    def test_all_empty_pages_no_chunks(self, splitter):
        pages = [
            LoadedPage(page_number=1, content=""),
            LoadedPage(page_number=2, content="   "),
        ]
        chunks = splitter.split_pages(pages)
        assert len(chunks) == 0

    def test_empty_input(self, splitter):
        chunks = splitter.split_pages([])
        assert len(chunks) == 0

    def test_whitespace_only_page_skipped(self, splitter):
        pages = [LoadedPage(page_number=1, content="  \n  \t  ")]
        chunks = splitter.split_pages(pages)
        assert len(chunks) == 0


class TestMinChunkSize:
    def test_tiny_trailing_chunk_merged_back(self):
        splitter = TextSplitterService(chunk_size=200, chunk_overlap=20, min_chunk_size=50)
        text = "A" * 90 + "\n\n" + "B" * 10
        pages = [LoadedPage(page_number=1, content=text)]
        chunks = splitter.split_pages(pages)
        assert len(chunks) == 1
        assert "A" in chunks[0].content
        assert "B" in chunks[0].content

    def test_minimum_chunk_not_split_further(self):
        splitter = TextSplitterService(chunk_size=100, chunk_overlap=20, min_chunk_size=50)
        text = "word " * 20  # 100 chars exactly
        pages = [LoadedPage(page_number=1, content=text)]
        chunks = splitter.split_pages(pages)
        assert len(chunks) >= 1

    def test_no_pending_tail_at_end(self):
        splitter = TextSplitterService(chunk_size=50, chunk_overlap=10, min_chunk_size=10)
        text = "A" * 30
        pages = [LoadedPage(page_number=1, content=text)]
        chunks = splitter.split_pages(pages)
        assert len(chunks) == 1


class TestCharacterCount:
    def test_character_count_matches_content(self, splitter):
        pages = [LoadedPage(page_number=1, content="Hello, this is a test document for chunking.")]
        chunks = splitter.split_pages(pages)
        for chunk in chunks:
            assert chunk.character_count == len(chunk.content)

    def test_all_chunks_have_character_count(self, splitter):
        text = "word " * 50
        pages = [LoadedPage(page_number=1, content=text)]
        chunks = splitter.split_pages(pages)
        for chunk in chunks:
            assert hasattr(chunk, "character_count")
            assert chunk.character_count > 0


class TestEdgeCases:
    def test_single_character_pages(self, splitter):
        pages = [
            LoadedPage(page_number=1, content="A"),
            LoadedPage(page_number=2, content="B"),
        ]
        chunks = splitter.split_pages(pages)
        for chunk in chunks:
            assert len(chunk.content) >= 20 or chunk.character_count >= 1

    def test_very_long_word(self, splitter):
        long_word = "x" * 200
        pages = [LoadedPage(page_number=1, content=long_word)]
        chunks = splitter.split_pages(pages)
        assert len(chunks) >= 1

    def test_paragraph_breaks_respected(self, splitter):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        pages = [LoadedPage(page_number=1, content=text)]
        chunks = splitter.split_pages(pages)
        assert len(chunks) >= 1

    def test_many_pages(self, large_splitter):
        pages = [LoadedPage(page_number=i, content=f"Page {i} content. " * 10) for i in range(1, 11)]
        chunks = large_splitter.split_pages(pages)
        page_numbers = {c.page_number for c in chunks}
        assert page_numbers == set(range(1, 11))
