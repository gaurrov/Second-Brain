"""
CSV text extraction. Reads CSV rows and produces a single LoadedPage
with rows formatted as structured text (column headers + pipe-separated
values) so the downstream chunker treats each row group as contextually
related content.
"""
import csv
import io
import logging
from pathlib import Path

from src.core.exceptions import TextExtractionException
from src.rag.loaders.base_loader import LoadedPage

logger = logging.getLogger("second_brain.loaders.csv")


class CSVLoader:
    def load(self, file_path: Path) -> list[LoadedPage]:
        if not file_path.exists():
            raise TextExtractionException(f"CSV file not found: {file_path}")
        if not file_path.is_file():
            raise TextExtractionException(f"Path is not a file: {file_path}")

        raw_bytes = file_path.read_bytes()
        if not raw_bytes:
            raise TextExtractionException("The uploaded CSV file is empty.")

        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1")

        if not text.strip():
            raise TextExtractionException("The uploaded CSV file is empty after decoding.")

        reader = csv.reader(io.StringIO(text))
        try:
            rows = list(reader)
        except csv.Error as exc:
            raise TextExtractionException(f"Could not parse CSV: {exc}") from exc

        if not rows:
            raise TextExtractionException("CSV file contains no rows.")

        headers = rows[0] if rows else []
        if not any(h.strip() for h in headers):
            raise TextExtractionException("CSV file has no column headers.")

        lines: list[str] = []
        for row_num, row in enumerate(rows[1:], start=2):
            pairs = []
            for i, value in enumerate(row):
                value = value.strip()
                if not value:
                    continue
                header = headers[i].strip() if i < len(headers) and headers[i].strip() else f"col_{i}"
                pairs.append(f"{header}: {value}")
            if pairs:
                lines.append(" | ".join(pairs))

        if not lines:
            raise TextExtractionException("CSV file contains no data rows.")

        content = f"Headers: {', '.join(h.strip() for h in headers if h.strip())}\n\n" + "\n".join(lines)

        logger.debug("CSV extracted %d data rows with %d columns", len(lines), len(headers))

        return [LoadedPage(page_number=None, content=content)]
