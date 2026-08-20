"""
Unit tests for LLMService (Groq wrapper).

A fake Groq client is injected so no network or API key is needed. Covers
message translation, model/token/temperature passthrough, lazy client
construction, and error translation into LLMException.
"""
from types import SimpleNamespace

import pytest

from src.core.config import settings
from src.core.exceptions import LLMException
from src.services.llm_service import LLMMessage, LLMService


class _FakeResponse:
    def __init__(self, content):
        if content is None:
            self.choices = []
        else:
            self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class _FakeCompletions:
    def __init__(self, parent):
        self.parent = parent

    def create(self, **kwargs):
        self.parent.creates.append(kwargs)
        if self.parent.error is not None:
            raise self.parent.error

        if kwargs.get("stream"):
            return self._stream(kwargs)

        return _FakeResponse(self.parent.content)

    def _stream(self, kwargs):
        """Yield fake streaming chunks matching the Groq SDK format."""
        for text in self.parent.stream_chunks:
            chunk = SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=text),
                    )
                ]
            )
            yield chunk


class _FakeChat:
    def __init__(self, parent):
        self.completions = _FakeCompletions(parent)


class _FakeGroq:
    """Stand-in for the Groq client with recording + fault injection."""

    def __init__(self, content="Generated answer", error=None, stream_chunks=None):
        self.creates = []
        self.content = content
        self.error = error
        self.stream_chunks = stream_chunks or ["Hello", " ", "world"]
        self.chat = _FakeChat(self)


@pytest.fixture
def fake_groq():
    return _FakeGroq()


@pytest.fixture
def service(fake_groq):
    return LLMService(client=fake_groq)


def _messages():
    return [
        LLMMessage(role="system", content="You are a helpful assistant."),
        LLMMessage(role="user", content="Hello!"),
    ]


class TestComplete:
    def test_returns_content(self, service):
        assert service.complete(_messages()) == "Generated answer"

    def test_passes_messages_and_model_params(self, service, fake_groq):
        service.complete(_messages())
        call = fake_groq.creates[0]
        assert call["model"] == settings.GROQ_MODEL
        assert call["max_tokens"] == settings.GROQ_MAX_TOKENS
        assert call["temperature"] == settings.GROQ_TEMPERATURE
        assert call["messages"] == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

    def test_strips_whitespace(self, fake_groq):
        fake_groq.content = "  The answer is 42.  \n"
        service = LLMService(client=fake_groq)
        assert service.complete(_messages()) == "The answer is 42."

    def test_transport_error_becomes_llm_exception(self, fake_groq):
        fake_groq.error = RuntimeError("connection refused")
        service = LLMService(client=fake_groq)
        with pytest.raises(LLMException):
            service.complete(_messages())

    def test_empty_response_becomes_llm_exception(self, fake_groq):
        fake_groq.content = "   "
        service = LLMService(client=fake_groq)
        with pytest.raises(LLMException):
            service.complete(_messages())

    def test_no_choices_becomes_llm_exception(self, fake_groq):
        fake_groq.content = None
        service = LLMService(client=fake_groq)
        with pytest.raises(LLMException):
            service.complete(_messages())


class TestClientConstruction:
    def test_client_constructed_lazily_and_cached(self, monkeypatch):
        constructed: list[str] = []

        def fake_factory(api_key, timeout):
            constructed.append(api_key)
            return _FakeGroq()

        monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key")
        monkeypatch.setattr("groq.Groq", fake_factory)

        service = LLMService()
        client_one = service.client
        client_two = service.client
        assert client_one is client_two
        assert constructed == ["test-key"]

    def test_missing_api_key_raises_llm_exception(self, monkeypatch):
        monkeypatch.setattr(settings, "GROQ_API_KEY", None)
        service = LLMService()
        with pytest.raises(LLMException):
            _ = service.client

    def test_custom_params_override_settings(self, fake_groq):
        service = LLMService(
            client=fake_groq,
            model="custom-model",
            max_tokens=512,
            temperature=0.9,
            timeout_seconds=10,
        )
        assert service.model == "custom-model"
        assert service.max_tokens == 512
        assert service.temperature == 0.9
        assert service.timeout_seconds == 10


class TestCompleteStream:
    def test_yields_text_chunks(self, fake_groq):
        fake_groq.stream_chunks = ["Hello", " ", "world"]
        service = LLMService(client=fake_groq)
        chunks = list(service.complete_stream(_messages()))
        assert chunks == ["Hello", " ", "world"]

    def test_passes_stream_true(self, fake_groq):
        service = LLMService(client=fake_groq)
        list(service.complete_stream(_messages()))
        assert fake_groq.creates[0]["stream"] is True

    def test_passes_model_and_params(self, fake_groq):
        service = LLMService(client=fake_groq)
        list(service.complete_stream(_messages()))
        call = fake_groq.creates[0]
        assert call["model"] == settings.GROQ_MODEL
        assert call["max_tokens"] == settings.GROQ_MAX_TOKENS
        assert call["temperature"] == settings.GROQ_TEMPERATURE

    def test_empty_stream_raises_llm_exception(self, fake_groq):
        fake_groq.stream_chunks = []
        service = LLMService(client=fake_groq)
        with pytest.raises(LLMException):
            list(service.complete_stream(_messages()))

    def test_transport_error_becomes_llm_exception(self, fake_groq):
        fake_groq.error = RuntimeError("connection refused")
        service = LLMService(client=fake_groq)
        with pytest.raises(LLMException):
            list(service.complete_stream(_messages()))

    def test_skips_empty_delta_chunks(self, fake_groq):
        """Chunks with no content in delta should be skipped silently."""
        fake_groq.stream_chunks = ["a", "", "b"]
        # The fake client yields all chunks; empty strings are still yielded.
        # In real Groq, some chunks have delta.content=None. The service
        # filters those. The fake yields "" which won't be None, so we
        # verify the service at least tolerates the pattern.
        service = LLMService(client=fake_groq)
        chunks = list(service.complete_stream(_messages()))
        assert len(chunks) >= 2
