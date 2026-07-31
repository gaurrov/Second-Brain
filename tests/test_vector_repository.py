"""
Integration tests for the Qdrant vector layer.

These run against a real Qdrant engine in in-process local mode
(QdrantClient(":memory:")), so they exercise actual Qdrant semantics —
upsert, filters, similarity scoring, deletion, counts — without needing
Docker or a live server. They cover collection provisioning, batch
upserts, the full payload contract, multi-user isolation, and search.
"""
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.core.config import settings
from src.rag.splitters.text_splitter import TextChunk
from src.repositories.vector_repository import SearchResult, VectorRepository
from src.vectorstore.collection_manager import ensure_collection

DIM = settings.EMBEDDING_DIMENSION


def _vector(seed: int, dim: int = DIM) -> list[float]:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim)
    return (vector / np.linalg.norm(vector)).tolist()


def _chunk(index: int, page: int, content: str) -> TextChunk:
    return TextChunk(
        chunk_index=index,
        page_number=page,
        content=content,
        character_count=len(content),
    )


@pytest.fixture
def qdrant_client():
    client = QdrantClient(":memory:")
    ensure_collection(client)
    return client


@pytest.fixture
def repo(qdrant_client):
    return VectorRepository(qdrant_client)


def _scroll_all(client: QdrantClient) -> list:
    points, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    return points


class TestCollectionProvisioning:
    def test_collection_created_with_cosine_and_correct_dim(self, qdrant_client):
        info = qdrant_client.get_collection(settings.QDRANT_COLLECTION_NAME)
        vectors_config = info.config.params.vectors
        assert vectors_config.size == DIM
        assert vectors_config.distance.value == "Cosine"

    def test_ensure_collection_is_idempotent(self, qdrant_client):
        ensure_collection(qdrant_client)  # second call must not raise
        info = qdrant_client.get_collection(settings.QDRANT_COLLECTION_NAME)
        assert info.config.params.vectors.size == DIM

    def test_payload_indexes_registered_without_error(self, qdrant_client):
        # Local mode reports indexes as no-ops, but the provisioning path
        # must run cleanly for every indexed field.
        ensure_collection(qdrant_client)


class TestBatchUpsert:
    def test_upsert_writes_complete_payload(self, repo, qdrant_client):
        user_id = uuid.uuid4()
        document_id = uuid.uuid4()
        timestamp = "2026-07-31T10:00:00+00:00"
        chunks = [
            _chunk(0, 1, "First chunk of page one."),
            _chunk(1, 2, "Second chunk on page two."),
        ]
        embeddings = [_vector(1), _vector(2)]

        written = repo.upsert_chunks(
            user_id=user_id,
            document_id=document_id,
            filename="notes.txt",
            chunks=chunks,
            embeddings=embeddings,
            timestamp=timestamp,
        )

        assert written == 2
        points = _scroll_all(qdrant_client)
        assert len(points) == 2
        for point in points:
            payload = point.payload
            assert payload["user_id"] == str(user_id)
            assert payload["document_id"] == str(document_id)
            assert payload["filename"] == "notes.txt"
            assert payload["timestamp"] == timestamp
            assert payload["content"]
            assert isinstance(payload["chunk_index"], int)
            assert isinstance(payload["page_number"], int)
        page_numbers = {p.payload["page_number"] for p in points}
        chunk_indices = {p.payload["chunk_index"] for p in points}
        assert page_numbers == {1, 2}
        assert chunk_indices == {0, 1}

    def test_upsert_splits_into_batches(self, repo, qdrant_client, monkeypatch):
        monkeypatch.setattr(settings, "QDRANT_UPSERT_BATCH_SIZE", 2)
        user_id = uuid.uuid4()
        document_id = uuid.uuid4()
        chunks = [_chunk(i, 1, f"chunk {i}") for i in range(5)]
        embeddings = [_vector(i) for i in range(5)]

        written = repo.upsert_chunks(
            user_id=user_id,
            document_id=document_id,
            filename="big.txt",
            chunks=chunks,
            embeddings=embeddings,
        )

        assert written == 5
        assert repo.count_by_document(document_id, user_id) == 5
        assert len(_scroll_all(qdrant_client)) == 5

    def test_upsert_default_timestamp_is_utc_iso(self, repo, qdrant_client):
        repo.upsert_chunks(
            user_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            filename="x.txt",
            chunks=[_chunk(0, 1, "hello")],
            embeddings=[_vector(1)],
        )
        payload = _scroll_all(qdrant_client)[0].payload
        parsed = datetime.fromisoformat(payload["timestamp"])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timezone.utc.utcoffset(parsed)

    def test_upsert_empty_chunks_returns_zero(self, repo, qdrant_client):
        written = repo.upsert_chunks(
            user_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            filename="empty.txt",
            chunks=[],
            embeddings=[],
        )
        assert written == 0
        assert _scroll_all(qdrant_client) == []

    def test_upsert_length_mismatch_raises(self, repo):
        with pytest.raises(ValueError):
            repo.upsert_chunks(
                user_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                filename="bad.txt",
                chunks=[_chunk(0, 1, "a")],
                embeddings=[],
            )


class TestCountAndDelete:
    def test_count_by_document_is_scoped(self, repo, qdrant_client):
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()
        doc_a = uuid.uuid4()
        doc_b = uuid.uuid4()

        repo.upsert_chunks(user_a, doc_a, "a.txt", [_chunk(0, 1, "a0"), _chunk(1, 1, "a1")], [_vector(1), _vector(2)])
        repo.upsert_chunks(user_b, doc_b, "b.txt", [_chunk(0, 1, "b0")], [_vector(3)])

        assert repo.count_by_document(doc_a, user_a) == 2
        assert repo.count_by_document(doc_b, user_b) == 1
        # Cross-user: user B has no chunks in doc_a.
        assert repo.count_by_document(doc_a, user_b) == 0
        assert len(_scroll_all(qdrant_client)) == 3

    def test_delete_removes_only_target_document(self, repo, qdrant_client):
        user_id = uuid.uuid4()
        doc_a = uuid.uuid4()
        doc_b = uuid.uuid4()
        repo.upsert_chunks(user_id, doc_a, "a.txt", [_chunk(0, 1, "a0"), _chunk(1, 1, "a1")], [_vector(1), _vector(2)])
        repo.upsert_chunks(user_id, doc_b, "b.txt", [_chunk(0, 1, "b0")], [_vector(3)])

        repo.delete_by_document(doc_a, user_id)

        assert repo.count_by_document(doc_a, user_id) == 0
        assert repo.count_by_document(doc_b, user_id) == 1
        assert len(_scroll_all(qdrant_client)) == 1

    def test_delete_is_scoped_by_user(self, repo, qdrant_client):
        shared_doc = uuid.uuid4()
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()
        repo.upsert_chunks(user_a, shared_doc, "shared.txt", [_chunk(0, 1, "a chunk")], [_vector(1)])
        repo.upsert_chunks(user_b, shared_doc, "shared.txt", [_chunk(0, 1, "b chunk")], [_vector(2)])

        repo.delete_by_document(shared_doc, user_a)

        assert repo.count_by_document(shared_doc, user_a) == 0
        assert repo.count_by_document(shared_doc, user_b) == 1


class TestSearch:
    def _seed_two_users(self, repo):
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()
        doc_a = uuid.uuid4()
        doc_b = uuid.uuid4()
        repo.upsert_chunks(user_a, doc_a, "a.txt", [_chunk(0, 1, "alpha one")], [_vector(1)])
        repo.upsert_chunks(user_b, doc_b, "b.txt", [_chunk(0, 2, "beta two")], [_vector(99)])
        return user_a, user_b, doc_a, doc_b

    def test_search_returns_most_similar_first(self, repo, qdrant_client):
        user_id = uuid.uuid4()
        document_id = uuid.uuid4()
        v1, v2, v3 = _vector(1), _vector(2), _vector(3)
        repo.upsert_chunks(
            user_id, document_id, "d.txt",
            [_chunk(0, 1, "c0"), _chunk(1, 1, "c1"), _chunk(2, 1, "c2")],
            [v1, v2, v3],
        )

        results = repo.search(v2, user_id, limit=3)

        assert len(results) == 3
        assert results[0].score == pytest.approx(1.0, abs=1e-6)
        # The exact query vector must rank first.
        assert results[0].content == "c1"
        assert results[0].score >= results[1].score >= results[2].score

    def test_search_respects_limit(self, repo):
        user_id = uuid.uuid4()
        document_id = uuid.uuid4()
        repo.upsert_chunks(
            user_id, document_id, "d.txt",
            [_chunk(i, 1, f"c{i}") for i in range(5)],
            [_vector(i) for i in range(5)],
        )
        results = repo.search(_vector(1), user_id, limit=2)
        assert len(results) == 2

    def test_search_isolates_users(self, repo):
        user_a, user_b, _, _ = self._seed_two_users(repo)

        results_a = repo.search(_vector(1), user_a, limit=10)
        results_b = repo.search(_vector(1), user_b, limit=10)

        assert len(results_a) == 1
        assert results_a[0].user_id == str(user_a)
        assert len(results_b) == 1
        assert results_b[0].user_id == str(user_b)

    def test_search_filters_by_document(self, repo):
        user_a, _, doc_a, doc_b = self._seed_two_users(repo)

        results = repo.search(_vector(1), user_a, document_id=doc_a, limit=10)
        assert len(results) == 1
        assert results[0].document_id == str(doc_a)
        assert results[0].content == "alpha one"

        assert repo.search(_vector(1), user_a, document_id=doc_b, limit=10) == []

    def test_search_filters_by_page_number(self, repo):
        user_id = uuid.uuid4()
        document_id = uuid.uuid4()
        repo.upsert_chunks(
            user_id, document_id, "d.txt",
            [_chunk(0, 1, "page1"), _chunk(1, 3, "page3")],
            [_vector(1), _vector(2)],
        )

        results = repo.search(_vector(1), user_id, page_number=3, limit=10)
        assert len(results) == 1
        assert results[0].page_number == 3
        assert results[0].content == "page3"

    def test_search_score_threshold_filters_weak_matches(self, repo):
        user_id = uuid.uuid4()
        document_id = uuid.uuid4()
        repo.upsert_chunks(
            user_id, document_id, "d.txt",
            [_chunk(0, 1, "exact"), _chunk(1, 1, "unrelated")],
            [_vector(5), _vector(6)],
        )

        results = repo.search(_vector(5), user_id, limit=10, score_threshold=0.999)
        assert len(results) == 1
        assert results[0].content == "exact"

    def test_search_returns_typed_results(self, repo):
        user_a, _, doc_a, _ = self._seed_two_users(repo)
        results = repo.search(_vector(1), user_a, limit=10)
        result = results[0]
        assert isinstance(result, SearchResult)
        assert result.user_id == str(user_a)
        assert result.document_id == str(doc_a)
        assert result.filename == "a.txt"
        assert result.page_number == 1
        assert result.chunk_index == 0
        assert result.content == "alpha one"
        assert result.point_id
        assert isinstance(result.score, float)

    def test_search_unknown_user_returns_empty(self, repo):
        user_a, _, _, _ = self._seed_two_users(repo)
        stranger = uuid.uuid4()
        assert repo.search(_vector(1), stranger, limit=10) == []
