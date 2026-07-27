"""
Markdown text extraction. Markdown is plain text by nature, so this
loader reads the raw content directly. It strips code fences and image
references to produce cleaner chunks for embedding, while preserving
heading hierarchy and paragraph structure.
"""
import logging
import re
from pathlib import Path

from src.core.exceptions import TextExtractionException
from src.rag.loaders.base_loader import LoadedPage

logger = logging.getLogger("second_brain.loaders.markdown")

_FENCE_RE = re.compile(r"^```[\s\S]*?^```", re.MULTILINE)
_IMAGE_RE = re.compile(r"!\[.*?\]\(.*?\)")
_LINK_TEXT_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


class MarkdownLoader:
    def load(self, file_path: Path) -> list[LoadedPage]:
        if not file_path.exists():
            raise TextExtractionException(f"Markdown file not found: {file_path}")
        if not file_path.is_file():
            raise TextExtractionException(f"Path is not a file: {file_path}")

        raw_bytes = file_path.read_bytes()
        if not raw_bytes:
            raise TextExtractionException("The uploaded Markdown file is empty.")

        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1")

        if not text.strip():
            raise TextExtractionException("The uploaded Markdown file is empty after decoding.")

        text = _FENCE_RE.sub("", text)
        text = _IMAGE_RE.sub("", text)
        text = _LINK_TEXT_RE.sub(r"\1", text)
        text = _HTML_TAG_RE.sub("", text)

        if not text.strip():
            raise TextExtractionException(
                "No usable text content after stripping Markdown syntax."
            )

        return [LoadedPage(page_number=None, content=text)]
