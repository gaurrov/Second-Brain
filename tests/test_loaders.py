"""
Tests for document loaders (PDF, DOCX, TXT, Markdown, HTML, CSV).

Each loader is tested in isolation with synthetic files — no real
documents needed. Validates both happy paths and error handling
(missing files, empty files, binary detection, etc.).
"""
import csv
import io
import tempfile
from pathlib import Path

import pytest

from src.core.exceptions import TextExtractionException
from src.rag.loaders.base_loader import LoadedPage
from src.rag.loaders.csv_loader import CSVLoader
from src.rag.loaders.docx_loader import DOCXLoader
from src.rag.loaders.html_loader import HTMLLoader
from src.rag.loaders.markdown_loader import MarkdownLoader
from src.rag.loaders.txt_loader import TXTLoader


class TestTXTLoader:
    def setup_method(self):
        self.loader = TXTLoader()

    def test_load_utf8(self, tmp_path):
        file = tmp_path / "test.txt"
        file.write_text("Hello world\nThis is a test.", encoding="utf-8")
        pages = self.loader.load(file)
        assert len(pages) == 1
        assert pages[0].page_number is None
        assert "Hello world" in pages[0].content

    def test_load_latin1_fallback(self, tmp_path):
        file = tmp_path / "test.txt"
        file.write_bytes("caf\xe9 r\xe9sum\xe9".encode("latin-1"))
        pages = self.loader.load(file)
        assert len(pages) == 1
        assert "café" in pages[0].content

    def test_empty_file_raises(self, tmp_path):
        file = tmp_path / "empty.txt"
        file.write_bytes(b"")
        with pytest.raises(TextExtractionException, match="empty"):
            self.loader.load(file)

    def test_whitespace_only_raises(self, tmp_path):
        file = tmp_path / "whitespace.txt"
        file.write_text("   \n  \t  ")
        with pytest.raises(TextExtractionException, match="empty"):
            self.loader.load(file)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(TextExtractionException, match="not found"):
            self.loader.load(tmp_path / "nonexistent.txt")

    def test_binary_detection(self, tmp_path):
        file = tmp_path / "binary.bin"
        file.write_bytes(b"\x00" * 200 + b"not text" + b"\x00" * 200)
        with pytest.raises(TextExtractionException, match="binary"):
            self.loader.load(file)

    def test_preserves_content(self, tmp_path):
        content = "Line one\nLine two\nLine three"
        file = tmp_path / "multi.txt"
        file.write_text(content)
        pages = self.loader.load(file)
        assert "Line one" in pages[0].content
        assert "Line three" in pages[0].content


class TestMarkdownLoader:
    def setup_method(self):
        self.loader = MarkdownLoader()

    def test_load_basic_markdown(self, tmp_path):
        file = tmp_path / "doc.md"
        file.write_text("# Title\n\nSome content here.\n\n## Section\n\nMore content.")
        pages = self.loader.load(file)
        assert len(pages) == 1
        assert "# Title" in pages[0].content
        assert "Some content here" in pages[0].content

    def test_strips_code_fences(self, tmp_path):
        file = tmp_path / "doc.md"
        file.write_text("# Title\n\n```python\nprint('hello')\n```\n\nReal content.")
        pages = self.loader.load(file)
        content = pages[0].content
        assert "print" not in content
        assert "Real content" in content

    def test_strips_images(self, tmp_path):
        file = tmp_path / "doc.md"
        file.write_text("Before ![alt text](image.png) after.")
        pages = self.loader.load(file)
        assert "image.png" not in pages[0].content
        assert "Before" in pages[0].content
        assert "after" in pages[0].content

    def test_preserves_link_text(self, tmp_path):
        file = tmp_path / "doc.md"
        file.write_text("Click [here](https://example.com) for more.")
        pages = self.loader.load(file)
        assert "here" in pages[0].content
        assert "https://example.com" not in pages[0].content

    def test_empty_file_raises(self, tmp_path):
        file = tmp_path / "empty.md"
        file.write_bytes(b"")
        with pytest.raises(TextExtractionException, match="empty"):
            self.loader.load(file)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(TextExtractionException, match="not found"):
            self.loader.load(tmp_path / "nonexistent.md")

    def test_strips_html_tags(self, tmp_path):
        file = tmp_path / "doc.md"
        file.write_text("Text with <b>bold</b> and <i>italic</i> tags.")
        pages = self.loader.load(file)
        assert "<b>" not in pages[0].content
        assert "bold" in pages[0].content


class TestHTMLLoader:
    def setup_method(self):
        self.loader = HTMLLoader()

    def test_load_basic_html(self, tmp_path):
        file = tmp_path / "doc.html"
        file.write_text("<html><body><h1>Title</h1><p>Content.</p></body></html>")
        pages = self.loader.load(file)
        assert len(pages) == 1
        assert "Title" in pages[0].content
        assert "Content." in pages[0].content

    def test_strips_scripts(self, tmp_path):
        file = tmp_path / "doc.html"
        file.write_text("<html><body><p>Visible</p><script>var x = 1;</script><p>Also visible</p></body></html>")
        pages = self.loader.load(file)
        content = pages[0].content
        assert "Visible" in content
        assert "Also visible" in content
        assert "var x" not in content

    def test_strips_styles(self, tmp_path):
        file = tmp_path / "doc.html"
        file.write_text("<html><head><style>body { color: red; }</style></head><body><p>Content</p></body></html>")
        pages = self.loader.load(file)
        assert "color: red" not in pages[0].content
        assert "Content" in pages[0].content

    def test_empty_file_raises(self, tmp_path):
        file = tmp_path / "empty.html"
        file.write_bytes(b"")
        with pytest.raises(TextExtractionException, match="empty"):
            self.loader.load(file)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(TextExtractionException, match="not found"):
            self.loader.load(tmp_path / "nonexistent.html")

    def test_handles_malformed_html(self, tmp_path):
        file = tmp_path / "bad.html"
        file.write_text("<p>Unclosed paragraph<div>Nested</p></div>")
        pages = self.loader.load(file)
        assert "Unclosed paragraph" in pages[0].content

    def test_strips_noscript(self, tmp_path):
        file = tmp_path / "doc.html"
        file.write_text("<noscript>Please enable JS</noscript><p>Content</p>")
        pages = self.loader.load(file)
        assert "enable JS" not in pages[0].content
        assert "Content" in pages[0].content


class TestCSVLoader:
    def setup_method(self):
        self.loader = CSVLoader()

    def test_load_basic_csv(self, tmp_path):
        file = tmp_path / "data.csv"
        file.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA")
        pages = self.loader.load(file)
        assert len(pages) == 1
        assert "Headers: name, age, city" in pages[0].content
        assert "name: Alice" in pages[0].content
        assert "age: 30" in pages[0].content
        assert "city: NYC" in pages[0].content

    def test_load_csv_with_empty_values(self, tmp_path):
        file = tmp_path / "data.csv"
        file.write_text("a,b,c\n1,,3\n,,")
        pages = self.loader.load(file)
        content = pages[0].content
        assert "a: 1" in content
        assert "c: 3" in content

    def test_load_csv_no_data_rows(self, tmp_path):
        file = tmp_path / "header_only.csv"
        file.write_text("col1,col2,col3\n")
        with pytest.raises(TextExtractionException, match="no data rows"):
            self.loader.load(file)

    def test_empty_file_raises(self, tmp_path):
        file = tmp_path / "empty.csv"
        file.write_bytes(b"")
        with pytest.raises(TextExtractionException, match="empty"):
            self.loader.load(file)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(TextExtractionException, match="not found"):
            self.loader.load(tmp_path / "nonexistent.csv")

    def test_load_csv_latin1(self, tmp_path):
        file = tmp_path / "data.csv"
        file.write_bytes("name,city\nAlice,M\xfcnchen".encode("latin-1"))
        pages = self.loader.load(file)
        assert "München" in pages[0].content

    def test_single_column_csv(self, tmp_path):
        file = tmp_path / "single.csv"
        file.write_text("value\nfoo\nbar\nbaz")
        pages = self.loader.load(file)
        assert "value: foo" in pages[0].content
        assert "value: bar" in pages[0].content


class TestDOCXLoader:
    def setup_method(self):
        self.loader = DOCXLoader()

    def test_empty_file_raises(self, tmp_path):
        file = tmp_path / "empty.docx"
        file.write_bytes(b"")
        with pytest.raises(TextExtractionException, match="Could not open"):
            self.loader.load(file)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(TextExtractionException, match="not found"):
            self.loader.load(tmp_path / "nonexistent.docx")
