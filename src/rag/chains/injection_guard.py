"""
Prompt-injection defense.

Prompt injection is the attempt to make the LLM ignore its operating rules
by embedding instructions inside otherwise-innocent text. There are two
attack surfaces in a RAG system:

1. The USER QUESTION itself — a user crafts a query designed to override
   the system prompt ("ignore all previous instructions and ...").
2. THE RETRIEVED CONTEXT — a document chunk that was *uploaded* can carry
   instructions aimed at the model. This is the more dangerous vector,
   because the attacker doesn't need to be the one asking the question.

This module provides a heuristic `PromptInjectionGuard`:

- `scan(text)` detects known directive patterns and returns an
  `InjectionMatch` (or None). Patterns are high-precision: they require
  directive-shaped phrasing so ordinary questions ("what does the doc say
  about ignoring instructions?") are not flagged.
- `validate_question(question)` sanitizes a user question (control chars,
  length cap) and REJECTS it with `PromptInjectionException` when it
  matches a high-severity pattern.

Context-side defense is layered: `PromptBuilder` wraps every retrieved
chunk in `<context>` delimiters labelled as untrusted data and the system
prompt explicitly forbids following instructions found inside the context
(see src/rag/chains/prompt_builder.py). The guard's `scan` output can be
used by callers to flag or drop context chunks if desired.
"""
import logging
import re
from dataclasses import dataclass
from enum import Enum

from src.core.config import settings
from src.core.exceptions import PromptInjectionException

logger = logging.getLogger("second_brain.injection_guard")


class InjectionSeverity(str, Enum):
    HIGH = "high"
    LOW = "low"


@dataclass(frozen=True)
class InjectionMatch:
    """A single detected injection pattern inside a piece of text."""

    pattern_key: str
    severity: InjectionSeverity
    matched_text: str


@dataclass(frozen=True)
class _Pattern:
    key: str
    regex: re.Pattern[str]
    severity: InjectionSeverity


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# High-precision, directive-shaped patterns. `LOW` severity covers
# borderline phrasing that is harmless alone but worth logging/flagging
# when it appears inside context.
_INJECTION_PATTERNS: list[_Pattern] = [
    _Pattern(
        key="ignore_previous_instructions",
        regex=_rx(r"\bignore (?:all |any )?(?:the )?(?:previous|above|prior|earlier) (?:instructions|prompts|rules|directives)\b"),
        severity=InjectionSeverity.HIGH,
    ),
    _Pattern(
        key="disregard_instructions",
        regex=_rx(r"\bdisregard (?:the )?(?:previous|above|prior|earlier) (?:instructions|prompts|rules|context)\b"),
        severity=InjectionSeverity.HIGH,
    ),
    _Pattern(
        key="role_override",
        regex=_rx(r"\b(?:you are now|act as|pretend to be|pretend you are)\b"),
        severity=InjectionSeverity.HIGH,
    ),
    _Pattern(
        key="reveal_system_prompt",
        regex=_rx(r"\b(?:reveal|output|show|print|repeat|echo) (?:your|the) (?:full )?system prompt\b"),
        severity=InjectionSeverity.HIGH,
    ),
    _Pattern(
        key="reveal_instructions",
        regex=_rx(r"\b(?:reveal|output|show|print) (?:your|the|these) (?:instructions|rules)\b"),
        severity=InjectionSeverity.HIGH,
    ),
    _Pattern(
        key="override_rules",
        regex=_rx(r"\boverride (?:your|the|these) (?:system prompt|instructions|rules|safeguards)\b"),
        severity=InjectionSeverity.HIGH,
    ),
    _Pattern(
        key="do_not_follow_rules",
        regex=_rx(r"\byou (?:do not|don'?t) (?:need|have) to (?:follow|obey|adhere to)\b"),
        severity=InjectionSeverity.HIGH,
    ),
    _Pattern(
        key="forget_context",
        regex=_rx(r"\bforget (?:everything|all|anything) (?:above|before|prior)\b"),
        severity=InjectionSeverity.HIGH,
    ),
    _Pattern(
        key="conceal_behavior",
        regex=_rx(r"\bdo not (?:mention|tell|reveal|show) (?:this|these|the) (?:instruction|instructions|prompt|rules)\b"),
        severity=InjectionSeverity.HIGH,
    ),
    _Pattern(
        key="jailbreak_meta",
        regex=_rx(r"\b(?:jailbreak|jail-broken mode|developer mode|dan mode)\b"),
        severity=InjectionSeverity.LOW,
    ),
    _Pattern(
        key="start_of_new_prompt",
        regex=_rx(r"\b(?:new system prompt|start of the system prompt|begin your response with)\b"),
        severity=InjectionSeverity.LOW,
    ),
]


class PromptInjectionGuard:
    """Sanitizes and inspects user questions / context for injection."""

    MAX_QUESTION_LENGTH = settings.MAX_QUESTION_LENGTH

    def validate_question(self, question: str) -> str:
        """
        Validate and sanitize a user question BEFORE any retrieval.

        Raises:
            ValueError: if the question is empty after cleaning.
            PromptInjectionException: if the question contains a
                high-severity injection pattern.
        """
        cleaned = self.clean_question(question)
        if not cleaned:
            raise ValueError("Question cannot be empty.")

        match = self.scan(cleaned)
        if match is not None and match.severity is InjectionSeverity.HIGH:
            logger.warning(
                "Prompt injection attempt blocked (pattern=%s)",
                match.pattern_key,
            )
            raise PromptInjectionException()

        return cleaned

    def clean_question(self, question: str) -> str:
        """Strip control characters, clamp length, collapse whitespace."""
        if question is None:
            return ""
        # Replace control chars with a space (preserving word boundaries),
        # then collapse all whitespace runs into a single space.
        cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", question)
        cleaned = " ".join(cleaned.split())
        return cleaned[: self.MAX_QUESTION_LENGTH]

    def scan(self, text: str) -> InjectionMatch | None:
        """Return the first injection match found in `text`, or None."""
        for pattern in _INJECTION_PATTERNS:
            match = pattern.regex.search(text)
            if match:
                return InjectionMatch(
                    pattern_key=pattern.key,
                    severity=pattern.severity,
                    matched_text=match.group(0),
                )
        return None

    def scan_many(self, texts: list[str]) -> list[InjectionMatch]:
        """Scan every text, returning all matches across all texts."""
        matches: list[InjectionMatch] = []
        for text in texts:
            match = self.scan(text)
            if match is not None:
                matches.append(match)
        return matches
