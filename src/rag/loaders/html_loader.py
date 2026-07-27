"""
HTML text extraction. Strips tags to produce plain text while preserving
structural elements (headings, lists, paragraphs). Uses stdlib html.parser
rather than pulling in a heavy dependency like BeautifulSoup, keeping
the project's dependency footprint lean.
"""
import logging
import re
from html.parser import HTMLParser
from pathlib import Path

from src.core.exceptions import TextExtractionException
from src.rag.loaders.base_loader import LoadedPage

logger = logging.getLogger("second_brain.loaders.html")


class _HTMLTextExtractor(HTMLParser):
    """
    Lightweight HTML-to-text converter. Collects text content from block
    elements and inserts newlines at natural boundaries (headings,
    paragraphs, list items, breaks).
    """

    _BLOCK_TAGS = frozenset({
        "p", "div", "section", "article", "header", "footer", "main",
        "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "blockquote",
        "pre", "br", "hr",
    })

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip_depth += 1
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


class HTMLLoader:
    def load(self, file_path: Path) -> list[LoadedPage]:
        if not file_path.exists():
            raise TextExtractionException(f"HTML file not found: {file_path}")
        if not file_path.is_file():
            raise TextExtractionException(f"Path is not a file: {file_path}")

        raw_bytes = file_path.read_bytes()
        if not raw_bytes:
            raise TextExtractionException("The uploaded HTML file is empty.")

        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1")

        if not text.strip():
            raise TextExtractionException("The uploaded HTML file is empty after decoding.")

        extractor = _HTMLTextExtractor()
        try:
            extractor.feed(text)
        except Exception as exc:
            raise TextExtractionException(f"Could not parse HTML content: {exc}") from exc

        content = extractor.get_text()
        if not content:
            raise TextExtractionException("No extractable text found in the HTML document.")

        return [LoadedPage(page_number=None, content=content)]
