"""
Plain text loader. Tries UTF-8 first (the overwhelmingly common case),
falling back to latin-1 (which never raises on arbitrary byte sequences)
so a non-UTF-8 text file doesn't hard-fail the whole upload.
"""
from pathlib import Path

from src.core.exceptions import TextExtractionException
from src.rag.loaders.base_loader import LoadedPage


class TXTLoader:
    def load(self, file_path: Path) -> list[LoadedPage]:
        raw_bytes = file_path.read_bytes()

        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1")

        if not text.strip():
            raise TextExtractionException("The uploaded .txt file is empty.")

        return [LoadedPage(page_number=None, content=text)]
