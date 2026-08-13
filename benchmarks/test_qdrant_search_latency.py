"""
Qdrant filtered-search latency benchmark.

Runs against the real Qdrant engine in embedded (in-memory) mode — the same
approach the vector repository integration tests use — so no Docker/server is
needed. Seeds a user's collection with a known set of vectors, then measures
user_id-filtered cosine search at a few result-set sizes.

Reported ops/s == searches/sec at the given top_k.

Run: pytest benchmarks/test_qdrant_search_latency.py --benchmark-only
"""
import uuid

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.core.config import settings
from src.rag.splitters.text_splitter import TextChunk
from src.repositories.vector_repository import VectorRepository
from src.vectorstore.collection_manager import ensure_collection

DIM = settings.EMBEDDING_DIMENSION
NUM_VECTORS_PER_USER = 2000


def _vector(seed: int, dim: int = DIM) -> list[float]:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim)
    return (vector / np.linalg.norm(vector)).tolist()


def _chunk(index: int, content: str) -> TextChunk:
    return TextChunk(
        chunk_index=index,
        page_number=1,
        content=content,
        character_count=len(content),
    )


@pytest.fixture(scope="module")
def repo() -> VectorRepository:
    client = QdrantClient(":memory:")
    ensure_collection(client)
    return VectorRepository(client)


@pytest.fixture(scope="module")
def seeded(repo) -> dict:
    """Two users, each with NUM_VECTORS_PER_USER distinct vectors."""
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    for user_id in (user_a, user_b):
        repo.upsert_chunks(
            user_id=user_id,
            document_id=uuid.uuid4(),
            filename="corpus.txt",
            chunks=[_chunk(i, f"content chunk number {i}") for i in range(NUM_VECTORS_PER_USER)],
            embeddings=[_vector(i) for i in range(NUM_VECTORS_PER_USER)],
        )
    return {"user_a": user_a, "user_b": user_b}


@pytest.mark.parametrize("top_k", [5, 20, 50])
def test_filtered_search_latency(benchmark, repo, seeded, top_k):
    query = _vector(12345)
    user_a = seeded["user_a"]

    def run():
        results = repo.search(query, user_a, limit=top_k)
        return results

    results = benchmark(run)
    assert len(results) == top_k
    benchmark.extra_info["top_k"] = top_k
    benchmark.extra_info["vectors_scanned_user"] = NUM_VECTORS_PER_USER
