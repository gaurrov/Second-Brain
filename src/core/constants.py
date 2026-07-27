"""
Shared enums/constants for the document module.

Defined once here so the ORM model, Pydantic schemas, and service layer
all reference the same values instead of duplicating string literals.
"""
from enum import Enum


class FileType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Maps an accepted file extension to its FileType, and doubles as the
# allow-list for upload validation — any extension not in this dict is
# rejected outright.
ALLOWED_EXTENSIONS: dict[str, FileType] = {
    ".pdf": FileType.PDF,
    ".docx": FileType.DOCX,
    ".txt": FileType.TXT,
    ".md": FileType.MARKDOWN,
    ".markdown": FileType.MARKDOWN,
    ".html": FileType.HTML,
    ".htm": FileType.HTML,
    ".csv": FileType.CSV,
}

# Accepted MIME types per extension, used as a second validation signal
# alongside the extension (defense against a mismatched/spoofed
# Content-Type or a renamed file extension).
ALLOWED_CONTENT_TYPES: dict[FileType, set[str]] = {
    FileType.PDF: {"application/pdf"},
    FileType.DOCX: {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/octet-stream",
    },
    FileType.TXT: {"text/plain", "application/octet-stream"},
    FileType.MARKDOWN: {"text/markdown", "text/x-markdown", "text/plain", "application/octet-stream"},
    FileType.HTML: {
        "text/html",
        "application/xhtml+xml",
        "application/octet-stream",
    },
    FileType.CSV: {"text/csv", "application/csv", "text/plain", "application/octet-stream"},
}

# Magic bytes for file-type validation (first N bytes of a file).
# Used as a defense-in-depth check alongside extension + MIME type to
# catch mismatched/spoofed Content-Type headers or renamed extensions.
MAGIC_BYTES: dict[FileType, list[bytes]] = {
    FileType.PDF: [b"%PDF"],
    FileType.DOCX: [b"PK\x03\x04"],
    FileType.TXT: [],  # text/plain has no reliable magic bytes
    FileType.MARKDOWN: [],  # plain text, no magic bytes
    FileType.HTML: [b"<!DOCTYPE", b"<html", b"<!doctype", b"<HTML"],
    FileType.CSV: [],  # plain text, no magic bytes
}

# Maximum number of bytes to read for magic-byte detection.
MAGIC_BYTES_READ_SIZE = 8

# Ingestion pipeline status labels (for structured logging).
PIPELINE_STEPS = [
    "validate",
    "extract",
    "clean",
    "chunk",
    "embed",
    "store",
    "finalize",
]
