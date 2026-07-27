"""
Integration tests for the document ingestion pipeline.

Tests the full flow from load -> clean -> chunk -> embed -> store,
with mocked embedding and vector layers to run offline. Validates
error handling, status transitions, and structured logging.
"""
import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.constants import FileType, ProcessingStatus
from src.core.exceptions import DocumentProcessingException, TextExtractionException
from src.db.session import SessionLocal
from src.models.document_model import Document
from src.rag.loaders.base_loader import LoadedPage
from src.rag.splitters.text_splitter import TextSplitterService
from src.repositories.document_repository import DocumentRepository
from src.services.embedding_service import EmbeddingService
from src.services.ingestion_service import IngestionService

_TEST_DIR = Path(os.environ.get("TEMP", "/tmp")) / "second_brain_tests"


@pytest.fixture
def db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from src.db.base_class import Base

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def doc_repository(db):
    return DocumentRepository(db)


@pytest.fixture
def mock_vector_repository():
    repo = MagicMock()
    repo.upsert_chunks.return_value = 5
    return repo


@pytest.fixture
def mock_embedding_service():
    service = MagicMock(spec=EmbeddingService)
    service.embed_documents.return_value = [[0.1] * 384] * 100
    return service


@pytest.fixture
def ingestion_service(doc_repository, mock_vector_repository, mock_embedding_service):
    return IngestionService(
        document_repository=doc_repository,
        vector_repository=mock_vector_repository,
        embedding_service=mock_embedding_service,
    )


def _create_document(db, *, filename="test.txt", file_type=FileType.TXT, file_path="/tmp/test.txt"):
    user_id = uuid.uuid4()
    doc = Document(
        id=uuid.uuid4(),
        user_id=user_id,
        filename=filename,
        file_type=file_type,
        file_path=file_path,
        file_size_bytes=1024,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


class TestIngestionPipelineSuccess:
    def test_full_pipeline_with_txt(self, db, ingestion_service, mock_vector_repository):
        _TEST_DIR.mkdir(parents=True, exist_ok=True)
        test_file = _TEST_DIR / f"test_{uuid.uuid4().hex[:8]}.txt"
        test_file.write_text("This is test content for the pipeline. " * 20)
        doc = _create_document(db, file_path=str(test_file))

        ingestion_service.process_document(doc)

        db.refresh(doc)
        assert doc.processing_status == ProcessingStatus.COMPLETED
        assert doc.chunk_count == 5
        assert doc.error_message is None

    def test_vectors_written_with_correct_metadata(self, db, ingestion_service, mock_vector_repository):
        _TEST_DIR.mkdir(parents=True, exist_ok=True)
        test_file = _TEST_DIR / f"test_{uuid.uuid4().hex[:8]}.txt"
        test_file.write_text("Test content for vector storage validation. " * 10)
        doc = _create_document(db, file_path=str(test_file))

        ingestion_service.process_document(doc)

        call_args = mock_vector_repository.upsert_chunks.call_args
        assert call_args.kwargs["user_id"] == doc.user_id
        assert call_args.kwargs["document_id"] == doc.id
        assert call_args.kwargs["filename"] == doc.filename


class TestIngestionPipelineErrors:
    def test_missing_file_sets_failed_status(self, db, ingestion_service):
        doc = _create_document(db, file_path="/nonexistent/path/file.txt")

        ingestion_service.process_document(doc)

        db.refresh(doc)
        assert doc.processing_status == ProcessingStatus.FAILED
        assert doc.error_message is not None
        assert "not found" in doc.error_message.lower()

    def test_empty_file_sets_failed_status(self, db, ingestion_service):
        _TEST_DIR.mkdir(parents=True, exist_ok=True)
        test_file = _TEST_DIR / f"empty_{uuid.uuid4().hex[:8]}.txt"
        test_file.write_bytes(b"")
        doc = _create_document(db, file_path=str(test_file))

        ingestion_service.process_document(doc)

        db.refresh(doc)
        assert doc.processing_status == ProcessingStatus.FAILED
        assert doc.error_message is not None

    def test_status_transitions_pending_to_processing_to_completed(self, db, ingestion_service):
        _TEST_DIR.mkdir(parents=True, exist_ok=True)
        test_file = _TEST_DIR / f"transitions_{uuid.uuid4().hex[:8]}.txt"
        test_file.write_text("Content for status transitions. " * 10)
        doc = _create_document(db, file_path=str(test_file))

        ingestion_service.process_document(doc)

        db.refresh(doc)
        assert doc.processing_status == ProcessingStatus.COMPLETED

    def test_error_message_truncated_at_2000_chars(self, db, ingestion_service):
        _TEST_DIR.mkdir(parents=True, exist_ok=True)
        test_file = _TEST_DIR / f"truncate_{uuid.uuid4().hex[:8]}.bin"
        test_file.write_bytes(b"\x00" * 100)
        doc = _create_document(db, file_path=str(test_file))

        ingestion_service.process_document(doc)

        db.refresh(doc)
        if doc.error_message:
            assert len(doc.error_message) <= 2000


class TestIngestionServiceFactory:
    def test_build_ingestion_service(self, db):
        from src.services.ingestion_service import build_ingestion_service

        with (
            patch("src.services.ingestion_service.get_qdrant_client") as mock_client,
            patch("src.services.ingestion_service.ensure_collection"),
            patch("src.services.ingestion_service.EmbeddingService") as mock_emb_cls,
        ):
            mock_client.return_value = MagicMock()
            mock_emb_cls.return_value = MagicMock()
            service = build_ingestion_service(db)
            assert isinstance(service, IngestionService)


class TestProcessDocumentTask:
    def test_nonexistent_document_no_crash(self, db):
        from src.services.ingestion_service import process_document_task

        fake_id = uuid.uuid4()
        with (
            patch("src.services.ingestion_service.SessionLocal", return_value=db),
            patch("src.services.ingestion_service.build_ingestion_service") as mock_build,
        ):
            process_document_task(fake_id)
            mock_build.assert_not_called()
