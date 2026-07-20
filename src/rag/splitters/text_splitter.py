"""
Text splitting: turns cleaned per-page text into overlapping chunks
suitable for embedding. Built on LangChain's RecursiveCharacterTextSplitter
(tries to split on paragraph/sentence/word boundaries before falling back
to a hard character cut), so chunks stay semantically coherent rather
than being cut mid-sentence whenever possible.
"""
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import settings
from src.rag.cleaners.text_cleaner import clean_text
from src.rag.loaders.base_loader import LoadedPage


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    page_number: int | None
    content: str


class TextSplitterService:
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or settings.CHUNK_SIZE,
            chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split_pages(self, pages: list[LoadedPage]) -> list[TextChunk]:
        """
        Cleans and splits each page independently (so a chunk never spans
        a page boundary, keeping page-number metadata accurate), then
        assigns a single running chunk_index across the whole document.
        """
        chunks: list[TextChunk] = []
        running_index = 0

        for page in pages:
            cleaned = clean_text(page.content)
            if not cleaned:
                continue

            for piece in self._splitter.split_text(cleaned):
                piece = piece.strip()
                if not piece:
                    continue
                chunks.append(
                    TextChunk(chunk_index=running_index, page_number=page.page_number, content=piece)
                )
                running_index += 1

        return chunks
