"""
Common interface all document loaders implement, and the value object
they return. Keeping this format-agnostic lets `ingestion_service.py`
treat PDF/DOCX/TXT uniformly after loading.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class LoadedPage:
    """One logical page/section of raw extracted text."""
    page_number: int | None  # None for formats with no page concept (e.g. .txt)
    content: str


class DocumentLoader(Protocol):
    def load(self, file_path: Path) -> list[LoadedPage]:
        """Extract raw text from `file_path`, returning one LoadedPage per page/section."""
        ...
