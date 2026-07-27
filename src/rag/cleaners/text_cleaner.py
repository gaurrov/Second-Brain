"""
Text cleaning/normalization applied to raw extracted text before it is
chunked. Keeping this isolated from the loaders means the same cleaning
rules apply uniformly regardless of source format (PDF/DOCX/TXT/HTML/CSV),
and it can evolve independently (e.g. adding de-hyphenation for PDFs)
without touching loader code.

Cleaning pipeline (in order):
  1. Unicode normalization (NFKC)
  2. Ligature expansion (fi -> fi, fl -> fl, etc.)
  3. Control character removal
  4. PDF artifact cleanup (page numbers, headers/footers)
  5. De-hyphenation across line breaks
  6. Whitespace collapsing (spaces, blank lines)
  7. Final trim
"""
import re
import unicodedata

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200c\u200d\ufeff]")
_MULTIPLE_BLANK_LINES_RE = re.compile(r"\n{3,}")
_MULTIPLE_SPACES_RE = re.compile(r"[ \t]{2,}")
_HYPHENATED_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")

# Common ligatures produced by PDF extraction / font encoding.
# NFKC normalizes some of these, but not all — handle the rest.
_LIGATURE_MAP: dict[str, str] = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}

# Repeated page numbers / footers common in PDFs (e.g. "- 3 -", "3 / 10", "Page 3").
_PAGE_NUM_RE = re.compile(
    r"^\s*[-–—]?\s*\d+\s*(/\s*\d+)?\s*[-–—]?\s*$",
    re.MULTILINE,
)

# Lines that are just a single repeated character (common PDF artifacts).
_ARTIFACT_LINE_RE = re.compile(r"^\s*[\-=_*\.]{3,}\s*$", re.MULTILINE)


def clean_text(raw_text: str) -> str:
    """
    Applies, in order:
      1. Unicode normalization (NFKC) — collapses visually-identical
         characters (e.g. ligatures, full-width variants) to canonical form.
      2. Ligature expansion for any remaining ligature characters.
      3. Removal of non-printable control characters.
      4. PDF artifact cleanup (standalone page numbers, separator lines).
      5. De-hyphenation across line breaks ("exam-\nple" -> "example").
      6. Collapsing runs of whitespace / blank lines.
      7. Final trim.
    """
    text = unicodedata.normalize("NFKC", raw_text)

    for ligature, replacement in _LIGATURE_MAP.items():
        text = text.replace(ligature, replacement)

    text = _CONTROL_CHARS_RE.sub("", text)

    text = _PAGE_NUM_RE.sub("", text)
    text = _ARTIFACT_LINE_RE.sub("", text)

    text = _HYPHENATED_LINEBREAK_RE.sub(r"\1\2", text)

    text = _MULTIPLE_SPACES_RE.sub(" ", text)
    text = _MULTIPLE_BLANK_LINES_RE.sub("\n\n", text)

    return text.strip()
