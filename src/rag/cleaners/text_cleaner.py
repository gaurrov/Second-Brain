"""
Text cleaning/normalization applied to raw extracted text before it is
chunked. Keeping this isolated from the loaders means the same cleaning
rules apply uniformly regardless of source format (PDF/DOCX/TXT), and it
can evolve independently (e.g. adding de-hyphenation for PDFs) without
touching loader code.
"""
import re
import unicodedata

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTIPLE_BLANK_LINES_RE = re.compile(r"\n{3,}")
_MULTIPLE_SPACES_RE = re.compile(r"[ \t]{2,}")
_HYPHENATED_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")


def clean_text(raw_text: str) -> str:
    """
    Applies, in order:
      1. Unicode normalization (NFKC) — collapses visually-identical
         characters (e.g. ligatures, full-width variants) to a canonical form.
      2. Removal of non-printable control characters (common in PDF extraction).
      3. De-hyphenation across line breaks (PDF text often wraps "exam-\nple" -> "example").
      4. Collapsing runs of whitespace/blank lines.
      5. Trimming leading/trailing whitespace.
    """
    text = unicodedata.normalize("NFKC", raw_text)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _HYPHENATED_LINEBREAK_RE.sub(r"\1\2", text)
    text = _MULTIPLE_SPACES_RE.sub(" ", text)
    text = _MULTIPLE_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()
