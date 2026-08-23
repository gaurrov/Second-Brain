"""
Query routing layer.

Classifies each incoming user message into one of two routes:

- ``CHAT`` -- conversational reply, no document retrieval required.
- ``DOCUMENT`` -- the user is asking about their uploaded knowledge base;
  semantic search against the vector store is needed.

The router is a **pure function** of the message text and recent
conversation history.  It:

- never reads or writes user identity (``user_id``),
- never touches Qdrant, the database, or any external service,
- never determines authentication or authorization,
- and is safe to call before (or instead of) the expensive
  embed -> search -> rerank pipeline.

Routing strategy (conservative, errs toward DOCUMENT):
    1. If the message is pure small-talk (``is_conversational``), route
       to CHAT -- greetings never need document context.
    2. If the message contains an explicit document-reference signal
       (possessive + document-type, file extension, attribution phrase,
       "my uploaded ..."), route to DOCUMENT.
    3. If the message is a short, ambiguous follow-up (no standard
       question-word opener) and the most recent assistant response was
       grounded in documents (``retrieval_metadata`` is non-empty),
       route to DOCUMENT -- the user is continuing the document
       conversation.
    4. Otherwise, default to CHAT -- general knowledge questions and
       open-ended conversation do not require retrieval.

Security invariants:
    - ``user_id`` is never read, written, or passed.
    - ``search_query`` is sanitized (control chars removed, length-capped)
      but never validated against a user schema -- it is a plain string
      handed to the vector search layer, which enforces its own
      ``user_id`` scoping independently.
"""
import re
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from src.rag.chains.intent_classifier import is_conversational


class Route(str, Enum):
    """The two possible routing decisions."""

    CHAT = "chat"
    DOCUMENT = "document"


@dataclass(frozen=True)
class RoutingResult:
    """Typed output of :meth:`QueryRouter.route`."""

    route: Route
    search_query: str | None = None


# ---------------------------------------------------------------------------
# Document-reference heuristics
# ---------------------------------------------------------------------------

# Possessive + document-type ("my notes", "my runbook", "my uploaded report")
_POSSESSIVE_DOCUMENT = re.compile(
    r"\bmy\b.+\b(?:document|file|notes?|runbook|upload|report|summary|"
    r"transcript|minutes|presentation|slide|spreadsheet|handout|syllabus|"
    r"resume|portfolio|thesis|paper|article|draft|log|spec|pdf|docx?|xlsx?)\b",
    re.IGNORECASE,
)

# "I write/wrote" patterns ("What did I write about...", "What I wrote in...")
_I_WROTE = re.compile(
    r"\b(?:what\s+)?(?:I|i)\s+(?:wrote|write)\b",
    re.IGNORECASE,
)

# File-extension references ("architecture.pdf", "notes.md")
_FILE_EXTENSION = re.compile(
    r"\b[\w.-]+\.(?:pdf|docx?|xlsx?|csv|txt|md|markdown|html?|pptx?|json)\b",
    re.IGNORECASE,
)

# Attribution phrases ("according to ...", "as per ...")
_ATTRIBUTION = re.compile(
    r"\b(?:according to|as per|based on|cited in|mentioned in|from)\b",
    re.IGNORECASE,
)

# "my uploaded ..." (any trailing word)
_MY_UPLOADED = re.compile(r"\bmy\s+uploaded\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Question-word detection (for history-based routing)
# ---------------------------------------------------------------------------

_EXPLICIT_QUESTION = re.compile(
    r"^\s*(?:who|what|when|where|why|how|which|whose|is|are|was|were|"
    r"can|could|would|should|do|does|did|will|shall|has|have|had)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------

class QueryRouter:
    """Decides whether a message requires document retrieval.

    Stateless and side-effect free -- safe to call per-request with no
    setup.  Instantiate once and reuse, or call as a plain function.

    Usage::

        router = QueryRouter()
        result = router.route("What does my runbook say about CI?")
        assert result.route is Route.DOCUMENT
        assert result.search_query == "What does my runbook say about CI?"

        result = router.route("Hi")
        assert result.route is Route.CHAT
        assert result.search_query is None
    """

    _MAX_FOLLOWUP_WORDS = 8

    def route(
        self,
        message: str,
        history: Sequence | None = None,
    ) -> RoutingResult:
        """Classify *message* as CHAT or DOCUMENT.

        Parameters
        ----------
        message:
            The raw user message (not yet sanitized by the injection guard).
        history:
            Optional recent conversation turns.  Each element must have
            ``role`` (str) and ``content`` (str) attributes -- compatible
            with :class:`src.rag.chains.prompt_builder.HistoryItem` and
            ORM :class:`Message` objects.

        Returns
        -------
        RoutingResult
            ``route`` is ``Route.CHAT`` or ``Route.DOCUMENT``.
            ``search_query`` is ``None`` for CHAT, and a sanitized copy
            of the message for DOCUMENT.
        """
        cleaned = message.strip()
        if not cleaned:
            return RoutingResult(route=Route.CHAT)

        # 1. Pure small-talk -> CHAT
        if is_conversational(cleaned):
            return RoutingResult(route=Route.CHAT)

        # 2. Explicit document signals -> DOCUMENT
        if self._has_document_signal(cleaned):
            return RoutingResult(
                route=Route.DOCUMENT,
                search_query=self._sanitize_query(cleaned),
            )

        # 3. History-aware: ambiguous follow-up after a document answer -> DOCUMENT
        if self._history_implies_document(cleaned, history):
            return RoutingResult(
                route=Route.DOCUMENT,
                search_query=self._sanitize_query(cleaned),
            )

        # 4. Default -> CHAT
        return RoutingResult(route=Route.CHAT)

    # ------------------------------------------------------------------
    # Document-signal detection
    # ------------------------------------------------------------------

    @staticmethod
    def _has_document_signal(text: str) -> bool:
        """Return True if *text* contains an explicit document reference."""
        if _POSSESSIVE_DOCUMENT.search(text):
            return True
        if _FILE_EXTENSION.search(text):
            return True
        if _ATTRIBUTION.search(text):
            return True
        if _MY_UPLOADED.search(text):
            return True
        if _I_WROTE.search(text):
            return True
        return False

    # ------------------------------------------------------------------
    # History-aware routing
    # ------------------------------------------------------------------

    def _history_implies_document(
        self,
        message: str,
        history: Sequence | None,
    ) -> bool:
        """Heuristic: ambiguous follow-up after a document-grounded answer.

        Returns True only when ALL of:
          - *message* is short (<= _MAX_FOLLOWUP_WORDS words)
          - *message* is NOT a standard question-word opener
          - *history* is non-empty and the last assistant message carried
            ``retrieval_metadata`` (i.e. it was a document-grounded answer)
        """
        if not history:
            return False

        words = message.split()
        if len(words) > self._MAX_FOLLOWUP_WORDS:
            return False

        if _EXPLICIT_QUESTION.match(message):
            return False

        # Walk backwards to find the most recent assistant message.
        for turn in reversed(history):
            role = getattr(turn, "role", None)
            if role is None:
                continue
            role_str = str(role).lower()
            # Handle both string values and enum values (MessageRole)
            if hasattr(role, "value"):
                role_str = str(role.value).lower()
            if role_str != "assistant":
                continue

            metadata = getattr(turn, "retrieval_metadata", None)
            if metadata:
                return True
            # Assistant message found but no metadata -> not document-grounded.
            return False

        return False

    # ------------------------------------------------------------------
    # Query sanitization
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_query(text: str) -> str:
        """Light sanitization of the search query.

        Removes control characters, collapses whitespace, and caps length.
        This is NOT security sanitization -- the vector search layer
        enforces its own scoping independently.
        """
        cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", text)
        cleaned = " ".join(cleaned.split())
        return cleaned[:500]
