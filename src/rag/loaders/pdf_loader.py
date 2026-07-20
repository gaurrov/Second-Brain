"""
PDF text extraction, one LoadedPage per PDF page (so downstream chunks
can carry accurate page-number metadata for citations later).
"""
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.core.exceptions import TextExtractionException
from src.rag.loaders.base_loader import LoadedPage


class PDFLoader:
    def load(self, file_path: Path) -> list[LoadedPage]:
        try:
            reader = PdfReader(str(file_path))
        except (PdfReadError, OSError) as exc:
            raise TextExtractionException(f"Could not open PDF file: {exc}") from exc

        if reader.is_encrypted:
            raise TextExtractionException("Cannot extract text from a password-protected PDF.")

        pages: list[LoadedPage] = []
        for index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # pypdf can raise various parsing errors per-page
                text = ""
            pages.append(LoadedPage(page_number=index, content=text))

        if not any(p.content.strip() for p in pages):
            raise TextExtractionException(
                "No extractable text found in this PDF (it may be a scanned/image-only document)."
            )

        return pages
