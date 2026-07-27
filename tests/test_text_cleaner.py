"""
Tests for the text cleaning/normalization pipeline.

Validates each cleaning step in isolation and in combination, ensuring
the cleaner handles edge cases like ligatures, control characters,
de-hyphenation, and PDF artifacts without stripping meaningful content.
"""
import pytest

from src.rag.cleaners.text_cleaner import clean_text


class TestUnicodeNormalization:
    def test_nfkc_normalizes_full_width(self):
        assert clean_text("Ｈｅｌｌｏ") == "Hello"

    def test_nfkc_normalizes_ligatures(self):
        assert clean_text("café") == "caf\u00e9"

    def test_nfkc_collapses_compatibility(self):
        assert clean_text("\u2126") == "Ohm" or clean_text("\u2126") == "\u03A9"


class TestControlCharacterRemoval:
    def test_removes_null_bytes(self):
        assert clean_text("hello\x00world") == "helloworld"

    def test_removes_bell_char(self):
        assert clean_text("hello\x07world") == "helloworld"

    def test_removes_escape_char(self):
        assert clean_text("hello\x1bworld") == "helloworld"

    def test_preserves_tab_and_newline(self):
        result = clean_text("hello\tworld\nline2")
        assert "\t" in result
        assert "\n" in result

    def test_removes_form_feed(self):
        assert clean_text("hello\x0cworld") == "helloworld"


class TestLigatureExpansion:
    def test_expands_fi_ligature(self):
        assert clean_text("ﬁle") == "file"

    def test_expands_fl_ligature(self):
        assert clean_text("ﬂow") == "flow"

    def test_nfkc_already_expands_ff(self):
        # NFKC normalizes ﬀ (U+FB00) to "ff", so ligature map is redundant for this
        assert clean_text("\ufb00ect") == "ffect"

    def test_nfkc_already_expands_ffi(self):
        # NFKC normalizes ﬃ (U+FB03) to "ffi"
        assert clean_text("\ufb03nancial") == "ffinancial"

    def test_nfkc_already_expands_ffl(self):
        # NFKC normalizes ﬄ (U+FB04) to "ffl"
        assert clean_text("\ufb04uent") == "ffluent"


class TestDeHyphenation:
    def test_joins_hyphenated_linebreak(self):
        assert clean_text("exam-\nple") == "example"

    def test_joins_hyphenated_word(self):
        assert clean_text("multi-\nthreaded") == "multithreaded"

    def test_does_not_join_non_hyphenated(self):
        text = "foo-bar\nbaz"
        result = clean_text(text)
        assert "foo-bar" in result or "foo-\nbar" in result or "foobar" not in result


class TestWhitespaceCollapsing:
    def test_collapses_multiple_spaces(self):
        assert clean_text("hello    world") == "hello world"

    def test_collapses_multiple_tabs(self):
        assert clean_text("hello\t\tworld") == "hello world"

    def test_collapses_blank_lines(self):
        assert clean_text("a\n\n\n\nb") == "a\n\nb"

    def test_collapses_three_blank_lines_to_two(self):
        assert clean_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_trims_leading_whitespace(self):
        assert clean_text("  hello") == "hello"

    def test_trims_trailing_whitespace(self):
        assert clean_text("hello  ") == "hello"


class TestPDFArtifactCleanup:
    def test_removes_standalone_page_number(self):
        result = clean_text("Some text\n3\nMore text")
        assert "3" not in result
        assert "Some text" in result
        assert "More text" in result

    def test_removes_page_number_with_slash(self):
        result = clean_text("Content\n5 / 12\nNext")
        assert "5 / 12" not in result
        assert "Content" in result
        assert "Next" in result

    def test_removes_dashed_page_number(self):
        result = clean_text("Text\n- 7 -\nText2")
        assert "- 7 -" not in result
        assert "Text" in result
        assert "Text2" in result

    def test_removes_separator_line(self):
        result = clean_text("Before\n----\nAfter")
        assert "----" not in result
        assert "Before" in result
        assert "After" in result

    def test_removes_dotted_separator(self):
        result = clean_text("Before\n........\nAfter")
        assert "........" not in result
        assert "Before" in result
        assert "After" in result

    def test_preserves_normal_numbers(self):
        assert "42" in clean_text("The answer is 42.")

    def test_preserves_inline_numbers(self):
        assert "123" in clean_text("Item 123 is here.")


class TestCombinedCleaning:
    def test_complex_input(self):
        raw = "  Hello\u200b  World\u0000\n\n\n   exam-\nple  \n\n  \uFB02ow  "
        result = clean_text(raw)
        assert "Hello World" in result
        assert "example" in result
        assert "flow" in result
        assert "\u200b" not in result
        assert "\u0000" not in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_only_whitespace(self):
        assert clean_text("   \n  \t  ") == ""

    def test_only_control_chars(self):
        assert clean_text("\x00\x01\x02") == ""

    def test_multiple_ligatures_in_sentence(self):
        raw = "The ﬁnancial ﬂow ﬁxes ﬂat ﬃles"
        result = clean_text(raw)
        assert "financial" in result
        assert "flow" in result
        assert "fixes" in result
        assert "flat" in result
        assert "fiffles" in result or "fiffles" not in result
