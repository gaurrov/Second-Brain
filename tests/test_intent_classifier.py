"""Tests for src/rag/chains/intent_classifier.py.

Every case from the specification is covered as an individual test,
plus edge cases for empty input, whitespace, mixed case, and trailing
punctuation variations. The tests are written to be immediately
readable and to serve as living documentation of the classifier's
contract.
"""
import pytest

from src.rag.chains.intent_classifier import is_conversational


# ---------------------------------------------------------------------------
# Should match (return True) — canonical small-talk phrases
# ---------------------------------------------------------------------------

class TestShouldMatch:
    """Messages that ARE pure small-talk — must return True."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "hi",
            "hello",
            "hey",
            "hi there",
            "hey there",
            "good morning",
            "good afternoon",
            "good evening",
            "thanks",
            "thank you",
            "bye",
            "goodbye",
            "see you",
            "how are you",
            "what's up",
            "ok",
            "okay",
            "cool",
            "got it",
            "sounds good",
            "nice",
            "great",
            "yo",
            "sup",
        ],
        ids=[
            "hi", "hello", "hey", "hi there", "hey there",
            "good_morning", "good_afternoon", "good_evening",
            "thanks", "thank you",
            "bye", "goodbye", "see you",
            "how are you", "what's up",
            "ok", "okay", "cool", "got it", "sounds good",
            "nice", "great", "yo", "sup",
        ],
    )
    def test_canonical_phrases(self, phrase: str) -> None:
        assert is_conversational(phrase) is True


# ---------------------------------------------------------------------------
# Must NOT match (return False) — real questions and content queries
# ---------------------------------------------------------------------------

class TestMustNotMatch:
    """Messages that are NOT pure small-talk — must return False."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "hi, what are the skills in my resume",
            "hello, can you summarize my notes on project X",
            "hey, list the key takeaways from my document",
            "how do I find my skills",
            "hi, how do I find my skills?",
            "what is the content of my resume",
            "can you explain the architecture",
            "tell me about the project",
            "summarize the meeting notes",
            "what are the action items",
            "list the deliverables from project X",
            "hi there, what does section 3 say",
            "good morning, summarize my notes",
            "hey, what's in my document",
            "thanks, but also can you find my resume",
            "ok, now search for project X",
        ],
        ids=[
            "hi_with_resume_question",
            "hello_with_summarize_question",
            "hey_with_list_question",
            "how_do_I_find",
            "hi_how_do_I_find_with_question_mark",
            "what_is_content",
            "explain_architecture",
            "tell_me_about_project",
            "summarize_meeting_notes",
            "what_are_action_items",
            "list_deliverables",
            "hi_there_what_does_section3",
            "good_morning_summarize",
            "hey_whats_in_document",
            "thanks_plus_resume_request",
            "ok_search_for_project",
        ],
    )
    def test_real_questions(self, phrase: str) -> None:
        assert is_conversational(phrase) is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_empty_string(self) -> None:
        assert is_conversational("") is False

    def test_whitespace_only(self) -> None:
        assert is_conversational("   ") is False

    def test_whitespace_only_tabs_and_newlines(self) -> None:
        assert is_conversational("\t\n  \r") is False


class TestMixedCase:
    """Case-insensitivity — must normalize before matching."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "Hi There!",
            "THANKS",
            "Hi",
            "HELLO",
            "Good Morning",
            "GOOD EVENING",
            "What's Up",
            "OK",
            "SOUNDS GOOD",
        ],
        ids=[
            "Hi_There_excl", "THANKS_upper", "Hi_cap",
            "HELLO_upper", "Good_Morning_cap", "GOOD_EVENING_upper",
            "Whats_Up_cap", "OK_upper", "SOUNDS_GOOD_upper",
        ],
    )
    def test_mixed_case_matches(self, phrase: str) -> None:
        assert is_conversational(phrase) is True


class TestTrailingPunctuation:
    """Trailing punctuation must be stripped before lookup."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "hi.",
            "hi!",
            "hi...",
            "hello!",
            "thanks!",
            "thanks.",
            "ok.",
            "okay!",
            "bye.",
            "bye!",
            "hey there!",
            "what's up?",
            "cool.",
            "nice!",
        ],
        ids=[
            "hi_dot", "hi_bang", "hi_ellipsis",
            "hello_bang", "thanks_bang", "thanks_dot",
            "ok_dot", "okay_bang", "bye_dot", "bye_bang",
            "hey_there_bang", "whats_up_question", "cool_dot", "nice_bang",
        ],
    )
    def test_trailing_punctuation_matches(self, phrase: str) -> None:
        assert is_conversational(phrase) is True


class TestQuestionsWithQuestionMarks:
    """Messages ending in '?' that are NOT small-talk must be rejected."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "what are the skills in my resume?",
            "can you summarize my notes?",
            "hi, how are you doing today?",
            "how do I find my skills?",
            "what does section 3 say?",
        ],
        ids=[
            "skills_question",
            "summarize_question",
            "hi_how_are_you_doing",
            "how_do_I_find",
            "what_does_section3",
        ],
    )
    def test_question_marks_reject(self, phrase: str) -> None:
        assert is_conversational(phrase) is False
