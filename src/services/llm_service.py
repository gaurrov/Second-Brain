"""
LLM service - Groq integration.

Thin, injectable wrapper around the Groq chat-completions API. The client
is constructed lazily (so importing this module never requires a key or a
network connection) and can be replaced with a fake in tests.

Resilience:
  - Transient failures (connection/timeout/5xx/rate-limit) are retried
    with exponential backoff via ``src.utils.retry``.
  - Every completion is timed and exported as a Prometheus histogram.
  - All API/transport errors are translated into the domain-level
    `LLMException` so the RAG service and HTTP layer never need to know
    about the Groq SDK's exception types.
"""
import logging
from dataclasses import dataclass
from typing import Protocol, Sequence

from src.core.config import settings
from src.core.exceptions import LLMException
from src.core.metrics import (
    llm_request_duration_seconds,
    llm_requests_total,
)
from src.utils.retry import retry

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


def _is_retryable_llm_error(exc: BaseException) -> bool:
    """Transient Groq/transport errors are retried; 4xx business errors are not."""
    if isinstance(exc, (TimeoutError, ConnectionError, OSError, BrokenPipeError)):
        return True
    # Match by class name to avoid importing the (heavy) groq SDK here.
    name = type(exc).__name__
    if name in {"APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError"}:
        return True
    if name == "APIStatusError":
        code = getattr(exc, "status_code", None)
        return code is None or code >= 500
    return False


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

            client = Groq(api_key=settings.GROQ_API_KEY, timeout=self.timeout_seconds)
            try:
                client.max_retries = 0  # our retry layer owns retries
            except AttributeError:
                pass
            self._client = client
        return self._client

    @retry(target="groq", retry_on=_is_retryable_llm_error)
    def complete(self, messages: Sequence[LLMMessage]) -> str:
        """
        Generate a single assistant turn for the given message list and
        return its text content. Raises LLMException on transport errors
        or an empty response.
        """
        try:
            with llm_request_duration_seconds.time():
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[message.to_dict() for message in messages],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
        except LLMException:
            llm_requests_total.labels(outcome="error").inc()
            raise
        except Exception as exc:  # noqa: BLE001 - translate any SDK/transport error
            llm_requests_total.labels(outcome="error").inc()
            logger.exception("Groq chat completion failed for model %s", self.model)
            raise LLMException(f"LLM generation failed: {exc}") from exc

        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            llm_requests_total.labels(outcome="empty").inc()
            logger.warning("Groq returned an empty completion for model %s", self.model)
            raise LLMException("The language model returned an empty response.")

        llm_requests_total.labels(outcome="success").inc()
        return content.strip()
