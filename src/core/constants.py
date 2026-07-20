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
}

# Accepted MIME types per extension, used as a second validation signal
# alongside the extension (defense against a mismatched/spoofed
# Content-Type or a renamed file extension).
ALLOWED_CONTENT_TYPES: dict[FileType, set[str]] = {
    FileType.PDF: {"application/pdf"},
    FileType.DOCX: {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/octet-stream",  # some clients send this generically
    },
    FileType.TXT: {"text/plain", "application/octet-stream"},
}
