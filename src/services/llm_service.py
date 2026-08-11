"""
LLM service - Groq integration.

Thin, injectable wrapper around the Groq chat-completions API. The client
is constructed lazily (so importing this module never requires a key or a
network connection) and can be replaced with a fake in tests.

All API/transport errors are translated into the domain-level
`LLMException` so the RAG service and HTTP layer never need to know about
the Groq SDK's exception types.
"""
import logging
from dataclasses import dataclass
from typing import Protocol, Sequence

from src.core.config import settings
from src.core.exceptions import LLMException

logger = logging.getLogger("second_brain.llm")


@dataclass(frozen=True)
class LLMMessage:
    """One message in a chat completion request."""

    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class ChatClient(Protocol):
    """Minimal surface of the Groq client used by LLMService."""

    def chat(self):
        ...


class LLMService:
    def __init__(
        self,
        client: ChatClient | None = None,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self._client = client
        self.model = model or settings.GROQ_MODEL
        self.max_tokens = max_tokens or settings.GROQ_MAX_TOKENS
        self.temperature = temperature if temperature is not None else settings.GROQ_TEMPERATURE
        self.timeout_seconds = timeout_seconds or settings.GROQ_TIMEOUT_SECONDS

    @property
    def client(self) -> ChatClient:
        """Lazily constructed Groq client (or the injected fake)."""
        if self._client is None:
            if not settings.GROQ_API_KEY:
                raise LLMException(
                    "GROQ_API_KEY is not configured; cannot generate answers."
                )
            from groq import Groq  # lazy import: no SDK required unless used

            self._client = Groq(api_key=settings.GROQ_API_KEY, timeout=self.timeout_seconds)
        return self._client

    def complete(self, messages: Sequence[LLMMessage]) -> str:
        """
        Generate a single assistant turn for the given message list and
        return its text content. Raises LLMException on transport errors
        or an empty response.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[message.to_dict() for message in messages],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except LLMException:
            raise
        except Exception as exc:  # noqa: BLE001 - translate any SDK/transport error
            logger.exception("Groq chat completion failed for model %s", self.model)
            raise LLMException(f"LLM generation failed: {exc}") from exc

        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            logger.warning("Groq returned an empty completion for model %s", self.model)
            raise LLMException("The language model returned an empty response.")

        return content.strip()
