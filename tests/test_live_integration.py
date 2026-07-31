"""
Live, opt-in end-to-end integration tests.

These exercise the REAL embedding model (settings.EMBEDDING_MODEL_NAME)
against a REAL Qdrant server, verifying the full embed -> store -> search
loop plus dimension matching and multi-user isolation. They are excluded
from the default suite — run them explicitly with:

    docker compose -f docker/docker-compose.yml up -d
    pytest --run-live tests/test_live_integration.py

If Qdrant is unreachable or model weights can't be loaded, the affected
tests skip rather than fail, so the suite is safe to run in environments
where the live stack isn't provisioned.
"""
import uuid

import pytest
from qdrant_client import QdrantClient

from src.core.config import settings
from src.rag.splitters.text_splitter import TextChunk
from src.repositories.vector_repository import VectorRepository
from src.services.embedding_service import EmbeddingService
from src.vectorstore.collection_manager import ensure_collection

pytestmark = pytest.mark.live


def _chunk(index: int, content: str) -> TextChunk:
    return TextChunk(
        chunk_index=index,
        page_number=1,
        content=content,
        character_count=len(content),
    )


@pytest.fixture(scope="module")
def qdrant_client() -> QdrantClient:
    client = QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        https=settings.QDRANT_USE_HTTPS,
        api_key=settings.QDRANT_API_KEY,
        timeout=5,
    )
    try:
        client.get_collections()
    except Exception as exc:  # noqa: BLE001 - opt-in live suite: skip if server is down
        pytest.skip(f"Qdrant unreachable at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}: {exc}")
    ensure_collection(client)
    return client


@pytest.fixture(scope="module")
def embedding_service(qdrant_client) -> EmbeddingService:
    try:
        # Bound network/hub timeouts so an unreachable registry or slow
        # first download skips instead of hanging the suite.
        import socket

        previous = socket.getdefaulttimeout()
        socket.setdefaulttimeout(30)
        try:
            service = EmbeddingService()
            service.embed_documents(["warmup sentence to load model weights"])
        finally:
            socket.setdefaulttimeout(previous)
        return service
    except Exception as exc:  # noqa: BLE001 - opt-in live suite: skip if weights unavailable
        pytest.skip(f"Embedding model unavailable: {exc}")


class TestEndToEndPipeline:
    def test_model_dimension_matches_collection(self, embedding_service, qdrant_client):
        info = qdrant_client.get_collection(settings.QDRANT_COLLECTION_NAME)
        assert info.config.params.vectors.size == embedding_service.dimension
        assert embedding_service.dimension == settings.EMBEDDING_DIMENSION

    def test_embed_store_search_delete_roundtrip(self, embedding_service, qdrant_client):
        user_id = uuid.uuid4()
        document_id = uuid.uuid4()
        repo = VectorRepository(qdrant_client)

        chunks = [
            _chunk(0, "The cat sat on the mat and purred loudly."),
            _chunk(1, "Quantum chromodynamics describes the strong force binding quarks."),
        ]
        embeddings = embedding_service.embed_documents([c.content for c in chunks])

        assert repo.upsert_chunks(user_id, document_id, "notes.txt", chunks, embeddings) == 2
        assert repo.count_by_document(document_id, user_id) == 2

        query = embedding_service.embed_query("a cat purring on a mat")
        results = repo.search(query, user_id, limit=1)
        assert len(results) == 1
        assert results[0].document_id == str(document_id)
        assert results[0].filename == "notes.txt"
        assert results[0].content == chunks[0].content

        # Retrieval is strictly user-scoped.
        stranger = uuid.uuid4()
        assert repo.search(query, stranger, limit=10) == []

        repo.delete_by_document(document_id, user_id)
        assert repo.count_by_document(document_id, user_id) == 0

    def test_payload_indexes_exist_on_server(self, qdrant_client):
        # Live servers expose the payload schema; local mode does not.
        info = qdrant_client.get_collection(settings.QDRANT_COLLECTION_NAME)
        schema = getattr(info.config, "payload_schema", None)
        if schema is None:
            pytest.skip("Qdrant client/version does not expose payload_schema")
        assert {"user_id", "document_id", "filename", "page_number", "chunk_index"} <= set(schema)
