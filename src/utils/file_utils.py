"""
File validation and storage utilities.

Pure I/O and validation helpers — no business logic, no DB access. Used
by `document_service.py` during the upload flow.
"""
import re
import uuid
from pathlib import Path

from fastapi import UploadFile

from src.core.config import settings
from src.core.constants import ALLOWED_CONTENT_TYPES, ALLOWED_EXTENSIONS, FileType
from src.core.exceptions import EmptyFileException, FileTooLargeException, UnsupportedFileTypeException

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.\-]")


def sanitize_filename(filename: str) -> str:
    """
    Strip any path components and replace unsafe characters, so a
    malicious filename (e.g. "../../etc/passwd") can never be used to
    escape the intended storage directory.
    """
    name = Path(filename).name  # drops any directory traversal components
    name = _UNSAFE_FILENAME_CHARS.sub("_", name)
    return name or "unnamed_file"


def get_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def validate_file_type(filename: str, content_type: str | None) -> FileType:
    """
    Validate the upload against the extension allow-list, and cross-check
    the declared Content-Type as a second signal. Raises
    UnsupportedFileTypeException if either check fails.
    """
    extension = get_extension(filename)
    file_type = ALLOWED_EXTENSIONS.get(extension)
    if file_type is None:
        raise UnsupportedFileTypeException(
            f"'{extension or 'unknown'}' is not supported. Allowed types: "
            f"{', '.join(sorted(e.lstrip('.') for e in ALLOWED_EXTENSIONS))}."
        )

    if content_type and content_type not in ALLOWED_CONTENT_TYPES[file_type]:
        raise UnsupportedFileTypeException(
            f"Content-Type '{content_type}' does not match a {file_type.value.upper()} file."
        )

    return file_type


def build_storage_path(user_id: uuid.UUID, document_id: uuid.UUID, filename: str) -> Path:
    """
    Builds the on-disk path for a stored document:
      {UPLOAD_DIR}/{user_id}/{document_id}/{sanitized_filename}

    Namespacing by user_id AND document_id gives physical isolation on
    disk in addition to the DB-level ownership checks — one user's files
    never share a directory with another's.
    """
    safe_name = sanitize_filename(filename)
    return Path(settings.UPLOAD_DIR) / str(user_id) / str(document_id) / safe_name


async def save_upload_file(upload_file: UploadFile, destination: Path) -> int:
    """
    Streams the upload to disk in chunks, enforcing MAX_UPLOAD_SIZE_MB
    without ever loading the whole file into memory at once. Returns the
    total number of bytes written. Raises FileTooLargeException /
    EmptyFileException as appropriate, cleaning up any partial file on
    failure.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    chunk_size = 1024 * 1024  # 1 MB
    total_bytes = 0

    try:
        with destination.open("wb") as out_file:
            while chunk := await upload_file.read(chunk_size):
                total_bytes += len(chunk)
                if total_bytes > settings.max_upload_size_bytes:
                    raise FileTooLargeException(
                        f"File exceeds the maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB} MB."
                    )
                out_file.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload_file.close()

    if total_bytes == 0:
        destination.unlink(missing_ok=True)
        raise EmptyFileException()

    return total_bytes


def delete_document_directory(user_id: uuid.UUID, document_id: uuid.UUID) -> None:
    """Removes the entire on-disk directory for a document (best-effort)."""
    import shutil

    doc_dir = Path(settings.UPLOAD_DIR) / str(user_id) / str(document_id)
    shutil.rmtree(doc_dir, ignore_errors=True)
