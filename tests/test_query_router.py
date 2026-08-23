"""
Unit tests for the query router.

Verifies that the router correctly classifies messages as CHAT or DOCUMENT,
handles edge cases, and respects conversation history for ambiguous follow-ups.
"""
import pytest

from src.rag.chains.query_router import QueryRouter, Route, RoutingResult


@pytest.fixture
def router():
    return QueryRouter()


# ---------------------------------------------------------------------------
# Chat route (no document signals)
# ---------------------------------------------------------------------------

class TestChatRoute:
    """Messages that should always route to CHAT."""

    @pytest.mark.parametrize("message", [
        "Hi",
        "Hello",
        "Hey",
        "Thanks",
        "How are you?",
        "What can you do?",
        "Tell me a joke",
        "What is Docker?",
        "Explain neural networks",
        "What is CQRS?",
        "How does encryption work?",
        "Tell me about microservices",
        "Summarize the concept of REST",
        "What are the SOLID principles?",
        "good morning",
        "yo",
        "got it",
        "see you",
    ])
    def test_general_knowledge_questions_route_to_chat(self, router, message):
        result = router.route(message)
        assert result.route is Route.CHAT
        assert result.search_query is None

    def test_empty_string_routes_to_chat(self, router):
        result = router.route("")
        assert result.route is Route.CHAT
        assert result.search_query is None

    def test_whitespace_only_routes_to_chat(self, router):
        result = router.route("   ")
        assert result.route is Route.CHAT
        assert result.search_query is None

    def test_pure_question_marks_route_to_chat(self, router):
        result = router.route("Why?")
        assert result.route is Route.CHAT
        assert result.search_query is None


# ---------------------------------------------------------------------------
# Document route (explicit signals)
# ---------------------------------------------------------------------------

class TestDocumentRoute:
    """Messages that should route to DOCUMENT due to explicit signals."""

    @pytest.mark.parametrize("message", [
        "What does my uploaded document say about CQRS?",
        "According to architecture.pdf, what is the proposed design?",
        "What did I write about RAG?",
        "Summarize my uploaded notes.",
        "Does my document mention Kubernetes?",
        "What does my microservices PDF say about Docker?",
        "Summarize my notes",
        "What is in my runbook?",
        "My report mentions scaling",
        "According to the slides, what was decided?",
        "Based on my thesis, what is the conclusion?",
        "What does my file say about authentication?",
        "Cite my presentation for the Q3 numbers",
        "What did I write in my report?",
        "According to my uploaded report, what are the findings?",
        "What does architecture.pdf say about scaling?",
        "What is in notes.md?",
        "Tell me what my document says",
    ])
    def test_explicit_document_signals_route_to_document(self, router, message):
        result = router.route(message)
        assert result.route is Route.DOCUMENT
        assert result.search_query is not None
        assert len(result.search_query) > 0

    def test_file_extension_detected(self, router):
        result = router.route("What does design-doc.docx say?")
        assert result.route is Route.DOCUMENT

    def test_my_uploaded_detected(self, router):
        result = router.route("Summarize my uploaded notes")
        assert result.route is Route.DOCUMENT

    def test_possessive_notes_detected(self, router):
        result = router.route("What are my notes about?")
        assert result.route is Route.DOCUMENT

    def test_possessive_runbook_detected(self, router):
        result = router.route("What is in my runbook?")
        assert result.route is Route.DOCUMENT

    def test_attribution_from_detected(self, router):
        result = router.route("From my document, what is the architecture?")
        assert result.route is Route.DOCUMENT


# ---------------------------------------------------------------------------
# Ambiguous follow-ups with history
# ---------------------------------------------------------------------------

class TestHistoryAwareRouting:
    """Ambiguous follow-ups should use conversation history to decide."""

    def _make_history(self, assistant_metadata=None):
        """Create a minimal history list with an assistant message."""
        class _Turn:
            def __init__(self, role, content, metadata=None):
                self.role = role
                self.content = content
                self.retrieval_metadata = metadata

        turns = []
        turns.append(_Turn("user", "What is CQRS?"))
        turns.append(_Turn("assistant", "CQRS separates reads and writes.", metadata=assistant_metadata))
        return turns

    def _make_metadata(self):
        return [{"document_id": "abc", "filename": "doc.pdf", "page_number": 1}]

    def test_bare_followup_after_document_answer_routes_to_document(self, router):
        history = self._make_history(assistant_metadata=self._make_metadata())
        result = router.route("and Docker?", history=history)
        assert result.route is Route.DOCUMENT
        assert result.search_query == "and Docker?"

    def test_bare_followup_no_metadata_routes_to_chat(self, router):
        history = self._make_history(assistant_metadata=None)
        result = router.route("and Docker?", history=history)
        assert result.route is Route.CHAT

    def test_explicit_question_always_routes_to_chat(self, router):
        history = self._make_history(assistant_metadata=self._make_metadata())
        result = router.route("What is Docker?", history=history)
        assert result.route is Route.CHAT
        assert result.search_query is None

    def test_how_does_it_work_routes_to_chat(self, router):
        history = self._make_history(assistant_metadata=self._make_metadata())
        result = router.route("How does it work?", history=history)
        assert result.route is Route.CHAT

    def test_why_does_it_do_that_routes_to_chat(self, router):
        history = self._make_history(assistant_metadata=self._make_metadata())
        result = router.route("Why does it do that?", history=history)
        assert result.route is Route.CHAT

    def test_long_followup_always_routes_to_chat(self, router):
        history = self._make_history(assistant_metadata=self._make_metadata())
        result = router.route("Can you explain in more detail how that works?", history=history)
        assert result.route is Route.CHAT

    def test_empty_history_followup_routes_to_chat(self, router):
        result = router.route("and it?", history=[])
        assert result.route is Route.CHAT

    def test_no_history_followup_routes_to_chat(self, router):
        result = router.route("and it?", history=None)
        assert result.route is Route.CHAT

    def test_bare_topic_after_document_routes_to_document(self, router):
        history = self._make_history(assistant_metadata=self._make_metadata())
        result = router.route("Docker?", history=history)
        assert result.route is Route.DOCUMENT

    def test_short_reference_with_the_after_document_routes_to_document(self, router):
        history = self._make_history(assistant_metadata=self._make_metadata())
        result = router.route("the API?", history=history)
        assert result.route is Route.DOCUMENT

    def test_explicit_question_with_how_after_document_stays_chat(self, router):
        history = self._make_history(assistant_metadata=self._make_metadata())
        result = router.route("how does Docker work?", history=history)
        assert result.route is Route.CHAT

    def test_user_role_turns_ignored_in_history(self, router):
        """Only assistant turns with metadata trigger history routing."""
        class _Turn:
            def __init__(self, role, content, metadata=None):
                self.role = role
                self.content = content
                self.retrieval_metadata = metadata

        history = [
            _Turn("user", "What is CQRS?"),
            _Turn("user", "And Docker?"),
        ]
        result = router.route("the API?", history=history)
        assert result.route is Route.CHAT

    def test_history_without_retrieval_metadata_attribute(self, router):
        """History turns without retrieval_metadata attribute are handled."""
        class _Turn:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        history = [
            _Turn("user", "What is CQRS?"),
            _Turn("assistant", "CQRS separates reads and writes."),
        ]
        result = router.route("and Docker?", history=history)
        assert result.route is Route.CHAT


# ---------------------------------------------------------------------------
# Explicit document signals bypass history check
# ---------------------------------------------------------------------------

class TestExplicitSignalBypassesHistory:
    """Explicit document signals route to DOCUMENT regardless of history."""

    def test_my_document_always_document(self, router):
        result = router.route("Does my architecture document mention it?")
        assert result.route is Route.DOCUMENT

    def test_file_extension_always_document(self, router):
        result = router.route("What about design.pdf?")
        assert result.route is Route.DOCUMENT


# ---------------------------------------------------------------------------
# Search query sanitization
# ---------------------------------------------------------------------------

class TestSearchQuerySanitization:
    """Search queries are sanitized before return."""

    def test_control_chars_removed(self, router):
        result = router.route("What does my document say about\x00CQRS?")
        assert result.route is Route.DOCUMENT
        assert "\x00" not in result.search_query

    def test_excess_whitespace_collapsed(self, router):
        result = router.route("What  does   my    document say?")
        assert result.route is Route.DOCUMENT
        assert "  " not in result.search_query

    def test_long_query_capped(self, router):
        message = "my document " + "word " * 200
        result = router.route(message)
        assert result.route is Route.DOCUMENT
        assert len(result.search_query) <= 500

    def test_chat_route_has_no_search_query(self, router):
        result = router.route("Hi")
        assert result.search_query is None


# ---------------------------------------------------------------------------
# RoutingResult structure
# ---------------------------------------------------------------------------

class TestRoutingResult:
    """RoutingResult is a properly typed dataclass."""

    def test_chat_result_fields(self):
        result = RoutingResult(route=Route.CHAT)
        assert result.route == "chat"
        assert result.search_query is None

    def test_document_result_fields(self):
        result = RoutingResult(route=Route.DOCUMENT, search_query="test query")
        assert result.route == "document"
        assert result.search_query == "test query"

    def test_route_enum_values(self):
        assert Route.CHAT.value == "chat"
        assert Route.DOCUMENT.value == "document"

    def test_result_is_frozen(self):
        result = RoutingResult(route=Route.CHAT)
        with pytest.raises(AttributeError):
            result.route = Route.DOCUMENT
