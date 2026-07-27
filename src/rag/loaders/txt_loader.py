"""
Plain text loader. Tries UTF-8 first (the overwhelmingly common case),
falling back to latin-1 (which never raises on arbitrary byte sequences)
so a non-UTF-8 text file doesn't hard-fail the whole upload. Also
handles binary detection — if the file looks like binary data rather
than text, we raise early instead of silently ingesting garbage.
"""
import logging
from pathlib import Path

from src.core.exceptions import TextExtractionException
from src.rag.loaders.base_loader import LoadedPage

logger = logging.getLogger("second_brain.loaders.txt")

_BINARY_THRESHOLD = 0.15  # >15% non-text bytes => treat as binary


def _looks_binary(data: bytes) -> bool:
    """
    Heuristic: sample the first 8 KB and count null bytes or
    non-printable characters. If the ratio exceeds the threshold,
    the file is likely binary and shouldn't be ingested as text.
    """
    if not data:
        return False
    sample = data[:8192]
    null_count = sample.count(b"\x00")
    non_text = sum(1 for b in sample if b < 0x09 or (0x0E <= b <= 0x1F and b != 0x1B))
    return (null_count + non_text) / len(sample) > _BINARY_THRESHOLD


class TXTLoader:
    def load(self, file_path: Path) -> list[LoadedPage]:
        if not file_path.exists():
            raise TextExtractionException(f"Text file not found: {file_path}")
        if not file_path.is_file():
            raise TextExtractionException(f"Path is not a file: {file_path}")

        raw_bytes = file_path.read_bytes()
        if not raw_bytes:
            raise TextExtractionException("The uploaded text file is empty.")

        if _looks_binary(raw_bytes):
            raise TextExtractionException(
                "File appears to be binary data, not a text file."
            )

        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            logger.debug("UTF-8 decode failed, falling back to latin-1 for %s", file_path.name)
            text = raw_bytes.decode("latin-1")

        if not text.strip():
            raise TextExtractionException("The uploaded text file is empty after decoding.")

        return [LoadedPage(page_number=None, content=text)]
