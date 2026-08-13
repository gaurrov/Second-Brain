"""
Embedding throughput benchmark (live: real model weights).

Measures how fast a batch of realistic chunks is embedded, both cold
(model must encode every chunk) and warm (the LRU text->vector cache
serves every chunk). Reported ops/s == batches/sec; chunks/sec = ops/s * BATCH_SIZE.

Run: pytest benchmarks/test_embedding_throughput.py --benchmark-only --run-live
"""
import pytest

from src.services.embedding_service import EmbeddingService

pytestmark = pytest.mark.live

BATCH_SIZE = 32
CHARS_PER_CHUNK = 400  # within the realistic 200-800 char chunk range

# Distinct, plausible prose blocks so every chunk is a cache miss when cold.
_BASE = (
    "The second brain methodology organizes knowledge into a graph of "
    "interconnected ideas rather than linear documents. By capturing notes, "
    "linking related concepts, and reviewing connections over time, an "
    "individual builds a durable external memory that compounds in value. "
    "Retrieval happens through semantic similarity instead of folder "
    "navigation, which surfaces relevant context even when the phrasing "
    "differs from the original note. "
)


def _chunk_batch(size: int, chars: int) -> list[str]:
    chunks = []
    for index in range(size):
        text = f"Section {index}. {_BASE * 3}"
        chunks.append(text[:chars])
    return chunks


@pytest.fixture(scope="module")
def embedding_service():
    import socket

    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(30)
    try:
        service = EmbeddingService()
        service.embed_documents(["warmup sentence to load model weights"])
        service.clear_cache()
        return service
    except Exception as exc:  # noqa: BLE001 - opt-in live suite: skip if weights unavailable
        pytest.skip(f"Embedding model unavailable: {exc}")
    finally:
        socket.setdefaulttimeout(previous)


def test_embed_throughput_cold(benchmark, embedding_service):
    chunks = _chunk_batch(BATCH_SIZE, CHARS_PER_CHUNK)
    embedding_service.clear_cache()

    def run():
        return embedding_service.embed_documents(chunks)

    benchmark(run)
    benchmark.extra_info["batch_size"] = BATCH_SIZE
    benchmark.extra_info["chunks_per_sec"] = round(BATCH_SIZE / benchmark.stats["mean"], 1)
    assert embedding_service.cache_size() == BATCH_SIZE


def test_embed_throughput_warm(benchmark, embedding_service):
    chunks = _chunk_batch(BATCH_SIZE, CHARS_PER_CHUNK)
    embedding_service.embed_documents(chunks)  # populate the L1 cache

    def run():
        return embedding_service.embed_documents(chunks)

    benchmark(run)
    benchmark.extra_info["batch_size"] = BATCH_SIZE
    benchmark.extra_info["chunks_per_sec"] = round(BATCH_SIZE / benchmark.stats["mean"], 1)
    assert embedding_service.cache_size() == BATCH_SIZE
