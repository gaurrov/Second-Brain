"""
Text splitting: turns cleaned per-page text into overlapping chunks
suitable for embedding.

This is a small, self-contained recursive character splitter — the same
algorithm LangChain's RecursiveCharacterTextSplitter uses (try splitting
on progressively finer separators, falling back to a hard cut), kept
in-house rather than pulled from `langchain-text-splitters`. That package
transitively drags in `langchain-core`, which hard-imports the compiled
`uuid_utils` extension for UUIDv7 generation — on locked-down Windows
environments with an Application Control Policy, that DLL gets blocked
and breaks the import chain entirely. Since we only ever used this one
splitting utility, vendoring it removes the dependency rather than
working around the block.

Improvements over the base implementation:
  - MIN_CHUNK_SIZE: rejects trivially short trailing chunks by merging
    them back into the previous chunk instead of emitting noise.
  - character_count on TextChunk: useful for cost estimation (token
    budgets correlate with character counts) and debugging.
"""
import logging
from dataclasses import dataclass

from src.core.config import settings
from src.rag.cleaners.text_cleaner import clean_text
from src.rag.loaders.base_loader import LoadedPage

logger = logging.getLogger("second_brain.text_splitter")

MIN_CHUNK_SIZE = 100


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    page_number: int | None
    content: str
    character_count: int


class _RecursiveCharacterSplitter:
    """
    Splits text into chunks of at most `chunk_size` characters, with
    `chunk_overlap` characters of overlap between consecutive chunks.
    Tries each separator in `separators` in order (paragraph breaks,
    then line breaks, then sentence breaks, then spaces), recursively
    splitting oversized pieces on the next separator down, and only
    falls back to a hard character cut if a piece has no separator at
    all left to split on.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> list[str]:
        pieces = self._split(text, self.separators)
        return self._merge_with_overlap(pieces)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text else []

        if not separators:
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        separator, remaining_separators = separators[0], separators[1:]

        if separator == "":
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = text.split(separator)
        results: list[str] = []
        for part in parts:
            if not part:
                continue
            if len(part) <= self.chunk_size:
                results.append(part)
            else:
                results.extend(self._split(part, remaining_separators))
        return results

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """Greedily packs split pieces back together up to chunk_size, carrying overlap forward."""
        if not pieces:
            return []

        merged: list[str] = []
        current = pieces[0]

        for piece in pieces[1:]:
            candidate = f"{current} {piece}" if current and piece else current + piece
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                merged.append(current)
                overlap_tail = current[-self.chunk_overlap :] if self.chunk_overlap else ""
                current = f"{overlap_tail} {piece}".strip() if overlap_tail else piece
        merged.append(current)

        return [chunk for chunk in merged if chunk.strip()]


class TextSplitterService:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_size: int = MIN_CHUNK_SIZE,
    ):
        self._splitter = _RecursiveCharacterSplitter(
            chunk_size=chunk_size or settings.CHUNK_SIZE,
            chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
        )
        self._min_chunk_size = min_chunk_size

    def split_pages(self, pages: list[LoadedPage]) -> list[TextChunk]:
        """
        Cleans and splits each page independently (so a chunk never spans
        a page boundary, keeping page-number metadata accurate), then
        assigns a single running chunk_index across the whole document.
        Trailing chunks shorter than min_chunk_size are merged back into
        the previous chunk to avoid emitting trivially short fragments.
        """
        chunks: list[TextChunk] = []
        running_index = 0

        for page in pages:
            cleaned = clean_text(page.content)
            if not cleaned:
                continue

            pieces = self._splitter.split_text(cleaned)
            pending_tail: str | None = None

            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    continue
                if pending_tail is not None:
                    piece = f"{pending_tail} {piece}".strip() if pending_tail else piece
                    pending_tail = None

                if len(piece) < self._min_chunk_size:
                    pending_tail = piece
                    continue

                chunks.append(
                    TextChunk(
                        chunk_index=running_index,
                        page_number=page.page_number,
                        content=piece,
                        character_count=len(piece),
                    )
                )
                running_index += 1

            if pending_tail is not None and pending_tail.strip():
                if chunks:
                    prev = chunks[-1]
                    merged = f"{prev.content} {pending_tail}".strip()
                    chunks[-1] = TextChunk(
                        chunk_index=prev.chunk_index,
                        page_number=prev.page_number,
                        content=merged,
                        character_count=len(merged),
                    )
                else:
                    chunks.append(
                        TextChunk(
                            chunk_index=running_index,
                            page_number=page.page_number,
                            content=pending_tail,
                            character_count=len(pending_tail),
                        )
                    )
                    running_index += 1

        logger.debug("Split %d pages into %d chunks", len(pages), len(chunks))
        return chunks