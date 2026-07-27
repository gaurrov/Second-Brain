"""
PDF text extraction, one LoadedPage per PDF page (so downstream chunks
can carry accurate page-number metadata for citations later).

Handles encrypted/scanned detection, per-page extraction errors, and
corrupt PDFs gracefully — a single failed page doesn't abort the
entire document.
"""
import logging
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.core.exceptions import TextExtractionException
from src.rag.loaders.base_loader import LoadedPage

logger = logging.getLogger("second_brain.loaders.pdf")


class PDFLoader:
    def load(self, file_path: Path) -> list[LoadedPage]:
        if not file_path.exists():
            raise TextExtractionException(f"PDF file not found: {file_path}")
        if not file_path.is_file():
            raise TextExtractionException(f"Path is not a file: {file_path}")

        try:
            reader = PdfReader(str(file_path))
        except (PdfReadError, OSError) as exc:
            raise TextExtractionException(f"Could not open PDF file: {exc}") from exc

        if reader.is_encrypted:
            raise TextExtractionException("Cannot extract text from a password-protected PDF.")

        page_count = len(reader.pages)
        if page_count == 0:
            raise TextExtractionException("PDF has no pages.")

        logger.debug("PDF has %d pages, extracting text", page_count)

        pages: list[LoadedPage] = []
        failed_pages = 0

        for index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                logger.warning("Failed to extract text from PDF page %d/%d: %s", index, page_count, exc)
                text = ""
                failed_pages += 1
            pages.append(LoadedPage(page_number=index, content=text))

        if failed_pages > 0:
            logger.warning(
                "PDF extraction: %d/%d pages failed to extract text", failed_pages, page_count
            )

        if not any(p.content.strip() for p in pages):
            raise TextExtractionException(
                "No extractable text found in this PDF (it may be a scanned/image-only document)."
            )

        return pages
