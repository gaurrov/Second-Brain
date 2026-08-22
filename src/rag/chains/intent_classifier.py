"""
Small-talk / greeting classifier for the RAG chat pipeline.

WHY THIS EXISTS:
  The RAG pipeline currently refuses any question that has no matching
  document content. Casual greetings like "hi" or "thanks" trigger that
  refusal, which feels broken. This module identifies genuinely
  conversational messages so a downstream service can reply with a
  friendly, non-RAG response instead.

DESIGN PRINCIPLE — ERR TOWARD FALSE:
  A false positive here would let a real document question bypass the
  refusal safeguard — e.g. "hi, summarize my notes on project X" would
  get a friendly greeting instead of a proper RAG answer. That is a
  worse failure mode than a false negative (which just means the user
  sees a refusal for "hi", annoying but harmless). Therefore:

    - Return True ONLY when the entire trimmed, lowercased message is
      a known small-talk phrase.
    - Return False for anything that contains a real question, looks
      like it's asking about content/documents/facts, or is ambiguous.
    - When in doubt, return False.

IMPLEMENTATION:
  A compact set of exact-match phrases and one lightweight regex for
  common punctuation/stripping variants. No ML, no LLM calls, no
  network requests — fast, deterministic, and easy to extend.
"""
import re

# Exact small-talk phrases, lowercased, with trailing punctuation stripped.
# This is the canonical set. A message must match one of these exactly
# (after normalization) to be classified as conversational.
_GREETINGS = {
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
}

# Phrases that accept trailing punctuation (e.g. "thanks!", "hi.", "ok...")
# are handled by stripping trailing punctuation before set lookup.
_TRAILING_PUNCTUATION = re.compile(r"[.!?…]+$")


def is_conversational(question: str) -> bool:
    """Return True if the message is purely a small-talk phrase.

    The check operates on the full trimmed, lowercased message after
    stripping trailing punctuation. The message must be an exact match
    against a known phrase — containment is not sufficient.

    Returns False for:
      - Empty or whitespace-only input
      - Messages that contain a real question or reference documents/facts
      - Any message longer than roughly 6–8 words
      - Ambiguous or unfamiliar phrases

    This is deliberately conservative. A downstream service uses
    ``False`` to mean "treat as a real question, apply the refusal
    safeguard if nothing matches in the vector store."
    """
    if not question:
        return False

    normalized = _TRAILING_PUNCTUATION.sub("", question.strip().lower())

    if not normalized:
        return False

    # Exact match after normalization
    if normalized in _GREETINGS:
        return True

    return False
