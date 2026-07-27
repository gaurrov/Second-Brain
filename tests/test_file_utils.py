"""
Tests for file validation and storage utilities.

Covers filename sanitization, extension/MIME validation, magic byte
detection, and storage path construction.
"""
import uuid

import pytest

from src.core.constants import FileType
from src.core.exceptions import (
    EmptyFileException,
    FileTooLargeException,
    UnsupportedFileTypeException,
)
from src.utils.file_utils import (
    build_storage_path,
    detect_file_type_by_magic,
    get_extension,
    sanitize_filename,
    validate_magic_bytes,
    validate_file_type,
)


class TestSanitizeFilename:
    def test_strips_path_traversal(self):
        assert sanitize_filename("../../etc/passwd") == "passwd"

    def test_replaces_unsafe_chars(self):
        result = sanitize_filename("file name (1).txt")
        assert result == "file_name__1.txt"

    def test_preserves_dots_and_hyphens(self):
        assert sanitize_filename("my-file.v2.txt") == "my-file.v2.txt"

    def test_empty_name_becomes_unnamed(self):
        assert sanitize_filename("") == "unnamed_file"

    def test_only_special_chars(self):
        result = sanitize_filename("@#$%^&()")
        assert result == "unnamed_file.txt" or result == "unnamed_file"

    def test_long_name_truncated(self):
        long_name = "a" * 300 + ".txt"
        result = sanitize_filename(long_name)
        assert len(result) <= 204

    def test_preserves_extension(self):
        assert sanitize_filename("data.csv").endswith(".csv")

    def test_strips_leading_trailing_dots(self):
        result = sanitize_filename("...file...")
        assert not result.startswith("...")
        assert not result.endswith("...")


class TestGetExtension:
    def test_lowercase(self):
        assert get_extension("file.PDF") == ".pdf"

    def test_no_extension(self):
        assert get_extension("Makefile") == ""

    def test_multiple_dots(self):
        assert get_extension("archive.tar.gz") == ".gz"

    def test_md_extension(self):
        assert get_extension("readme.MD") == ".md"

    def test_htm_extension(self):
        assert get_extension("index.htm") == ".htm"


class TestValidateFileType:
    def test_valid_pdf(self):
        assert validate_file_type("doc.pdf", "application/pdf") == FileType.PDF

    def test_valid_docx(self):
        assert validate_file_type("doc.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document") == FileType.DOCX

    def test_valid_txt(self):
        assert validate_file_type("doc.txt", "text/plain") == FileType.TXT

    def test_valid_markdown(self):
        assert validate_file_type("doc.md", "text/markdown") == FileType.MARKDOWN

    def test_valid_html(self):
        assert validate_file_type("doc.html", "text/html") == FileType.HTML

    def test_valid_csv(self):
        assert validate_file_type("data.csv", "text/csv") == FileType.CSV

    def test_rejects_unknown_extension(self):
        with pytest.raises(UnsupportedFileTypeException, match="not supported"):
            validate_file_type("file.exe", "application/octet-stream")

    def test_none_content_type_accepted(self):
        assert validate_file_type("doc.pdf", None) == FileType.PDF

    def test_octet_stream_accepted_for_docx(self):
        assert validate_file_type("doc.docx", "application/octet-stream") == FileType.DOCX

    def test_wrong_mime_type_rejected(self):
        with pytest.raises(UnsupportedFileTypeException, match="does not match"):
            validate_file_type("doc.pdf", "text/plain")


class TestMagicBytesDetection:
    def test_detect_pdf(self):
        assert detect_file_type_by_magic(b"%PDF-1.4stuff") == FileType.PDF

    def test_detect_docx(self):
        assert detect_file_type_by_magic(b"PK\x03\x04rest") == FileType.DOCX

    def test_detect_html_doctype(self):
        assert detect_file_type_by_magic(b"<!DOCTYPE html>") == FileType.HTML

    def test_detect_html_lowercase(self):
        assert detect_file_type_by_magic(b"<html>") == FileType.HTML

    def test_detect_html_uppercase(self):
        assert detect_file_type_by_magic(b"<HTML>") == FileType.HTML

    def test_no_match_for_text(self):
        assert detect_file_type_by_magic(b"Hello world") is None

    def test_no_match_for_empty(self):
        assert detect_file_type_by_magic(b"") is None

    def test_partial_match_not_enough(self):
        assert detect_file_type_by_magic(b"%") is None


class TestValidateMagicBytes:
    def test_pdf_magic_bytes_valid(self, tmp_path):
        file = tmp_path / "test.pdf"
        file.write_bytes(b"%PDF-1.4 content here")
        validate_magic_bytes(file, FileType.PDF)

    def test_pdf_magic_bytes_mismatch(self, tmp_path):
        file = tmp_path / "test.pdf"
        file.write_bytes(b"PK\x03\x04this is a docx")
        with pytest.raises(UnsupportedFileTypeException, match="does not match declared type"):
            validate_magic_bytes(file, FileType.PDF)

    def test_txt_no_magic_check(self, tmp_path):
        file = tmp_path / "test.txt"
        file.write_bytes(b"Hello")
        validate_magic_bytes(file, FileType.TXT)

    def test_missing_file_no_crash(self, tmp_path):
        validate_magic_bytes(tmp_path / "nonexistent.pdf", FileType.PDF)


class TestBuildStoragePath:
    def test_structure(self):
        user_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        path = build_storage_path(user_id, doc_id, "report.pdf")
        assert str(user_id) in str(path)
        assert str(doc_id) in str(path)
        assert "report.pdf" in str(path)

    def test_sanitizes_name(self):
        user_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        path = build_storage_path(user_id, doc_id, "../../etc/passwd")
        assert "passwd" in str(path)
        assert ".." not in str(path)

    def test_under_uploads_dir(self):
        user_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        path = build_storage_path(user_id, doc_id, "file.txt")
        assert str(path).startswith("uploads")
