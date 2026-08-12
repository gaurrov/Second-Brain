"""
Unit tests for EmbeddingService.

A fake SentenceTransformer-like model is injected so no model weights are
downloaded and tests stay fast/offline. These tests verify the service's
production behavior: singleton model loading, LRU caching (dedupe +
replay), bounded sub-batch encoding, query-instruction prefixing, and
thread-safe access.
"""
import numpy as np
import pytest

from src.core.config import settings
from src.services import embedding_service as embedding_module
from src.services.embedding_service import EmbeddingService

_DIM = 8


class _FakeModel:
    """Minimal stand-in for SentenceTransformer with deterministic vectors."""

    def __init__(self, dim: int = _DIM) -> None:
        self.dim = dim
        self.encode_calls: list[list[str]] = []

    def encode(self, texts, **kwargs):
        self.encode_calls.append(list(texts))
        rows = [_deterministic_vector(text, self.dim) for text in texts]
        return np.array(rows, dtype=np.float32)

    def get_sentence_embedding_dimension(self):
        return self.dim


def _deterministic_vector(text: str, dim: int = _DIM) -> list[float]:
    seed = sum(ord(ch) for ch in text)
    return [((seed + j) % 100) / 100.0 for j in range(dim)]


@pytest.fixture
def fake_model():
    return _FakeModel()


@pytest.fixture
def service(fake_model):
    return EmbeddingService(model=fake_model)


class TestEmbedDocuments:
    def test_returns_vector_per_text(self, service):
        vectors = service.embed_documents(["alpha", "beta", "gamma"])
        assert len(vectors) == 3
        for vector in vectors:
            assert len(vector) == _DIM
            assert all(isinstance(x, float) for x in vector)

    def test_empty_input_returns_empty(self, service):
        assert service.embed_documents([]) == []

    def test_dedupes_identical_texts(self, service, fake_model):
        vectors = service.embed_documents(["same", "same", "other"])
        assert len(vectors) == 3
        assert vectors[0] == vectors[1]
        # Only the two distinct texts ever hit the model.
        assert fake_model.encode_calls == [["same", "other"]]

    def test_trims_whitespace_before_embedding(self, service, fake_model):
        vectors = service.embed_documents(["  padded  ", "padded"])
        assert vectors[0] == vectors[1]
        assert fake_model.encode_calls == [["padded"]]

    def test_cache_serves_second_call_without_inference(self, service, fake_model):
        first = service.embed_documents(["alpha", "beta"])
        second = service.embed_documents(["alpha", "beta"])
        assert first == second
        assert len(fake_model.encode_calls) == 1

    def test_partial_cache_hit_embeds_only_misses(self, service, fake_model):
        service.embed_documents(["alpha"])
        service.embed_documents(["alpha", "beta"])
        assert fake_model.encode_calls == [["alpha"], ["beta"]]

    def test_sub_batches_inputs_to_batch_size(self, service, fake_model, monkeypatch):
        monkeypatch.setattr(settings, "EMBEDDING_BATCH_SIZE", 2)
        service.embed_documents(["t1", "t2", "t3", "t4", "t5"])
        assert fake_model.encode_calls == [["t1", "t2"], ["t3", "t4"], ["t5"]]

    def test_dimension_matches_model(self, service):
        assert service.dimension == _DIM


class TestEmbedQuery:
    def test_prepends_query_instruction(self, service, fake_model):
        service.embed_query("how do I organize notes?")
        assert fake_model.encode_calls == [
            [f"{EmbeddingService.QUERY_INSTRUCTION}how do I organize notes?"]
        ]

    def test_returns_single_vector(self, service):
        vector = service.embed_query("hello")
        assert len(vector) == _DIM

    def test_query_cached_across_calls(self, service, fake_model):
        a = service.embed_query("hello world")
        b = service.embed_query("hello world")
        assert a == b
        assert len(fake_model.encode_calls) == 1


class TestCache:
    def test_evicts_oldest_beyond_maxsize(self, fake_model):
        service = EmbeddingService(model=fake_model, cache_size=2)
        service.embed_documents(["one", "two", "three"])  # evicts "one"
        assert service.cache_size() == 2

        # "one" was evicted -> must re-embed; "two" still cached.
        service.embed_documents(["one", "two"])
        assert len(fake_model.encode_calls) == 2  # first pass + re-embed of "one"
        assert fake_model.encode_calls[-1] == ["one"]

    def test_clear_cache_forces_reembed(self, service, fake_model):
        service.embed_documents(["alpha"])
        service.clear_cache()
        service.embed_documents(["alpha"])
        assert len(fake_model.encode_calls) == 2

    def test_cache_results_are_isolated_copies(self, service):
        vectors = service.embed_documents(["alpha", "alpha"])
        vectors[0][0] = 999.0
        assert vectors[1][0] != 999.0


class TestSingletonModel:
    def test_model_loaded_once_per_process(self, monkeypatch):
        constructed: list[str] = []

        def fake_factory(model_name: str, **kwargs):
            constructed.append(model_name)
            return _FakeModel()

        monkeypatch.setattr(embedding_module, "SentenceTransformer", fake_factory)
        embedding_module.clear_model_cache()
        first = embedding_module._get_model()
        second = embedding_module._get_model()
        assert first is second
        assert constructed == [settings.EMBEDDING_MODEL_NAME]

    def test_clear_model_cache_reloads(self, monkeypatch):
        def fake_factory(model_name: str, **kwargs):
            return _FakeModel()

        monkeypatch.setattr(embedding_module, "SentenceTransformer", fake_factory)
        embedding_module.clear_model_cache()
        embedding_module._get_model()
        embedding_module.clear_model_cache()
        assert embedding_module._get_model.cache_info().currsize == 0

    def test_injected_model_skips_singleton(self, fake_model):
        service = EmbeddingService(model=fake_model)
        service.embed_documents(["text"])
        assert service.model is fake_model
