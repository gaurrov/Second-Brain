"""
DOCX text extraction using python-docx.

DOCX has no native "page" concept (pagination is a rendering-time
concern), so the whole document is returned as a single LoadedPage with
page_number=None. Paragraphs are joined with newlines to preserve
paragraph boundaries for the text splitter.
"""
from pathlib import Path

import docx
from docx.opc.exceptions import PackageNotFoundError

from src.core.exceptions import TextExtractionException
from src.rag.loaders.base_loader import LoadedPage


class DOCXLoader:
    def load(self, file_path: Path) -> list[LoadedPage]:
        try:
            document = docx.Document(str(file_path))
        except (PackageNotFoundError, OSError, KeyError) as exc:
            raise TextExtractionException(f"Could not open DOCX file: {exc}") from exc

        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]

        # Also pull text out of tables, which python-docx does not
        # include in `document.paragraphs`.
        for table in document.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)

        full_text = "\n".join(paragraphs)

        if not full_text.strip():
            raise TextExtractionException("No extractable text found in this DOCX file.")

        return [LoadedPage(page_number=None, content=full_text)]
