"""
Ingestion service — orchestrates the document processing pipeline:

    load (extract text) -> clean -> chunk -> embed -> store in Qdrant
                                                           |
                                                           v
                                          update Document row (Postgres)

This is invoked asynchronously after the upload endpoint returns (see
document_service.py), so a slow PDF never blocks the HTTP response.
It's a plain function (not tied to FastAPI) so it can be run from a
BackgroundTask today and from a Celery/RQ worker later with zero changes.
"""
import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from src.core.constants import ProcessingStatus
from src.core.exceptions import DocumentProcessingException
from src.db.session import SessionLocal
from src.models.document_model import Document
from src.rag.loaders.loader_factory import get_loader
from src.rag.splitters.text_splitter import TextSplitterService
from src.repositories.document_repository import DocumentRepository
from src.repositories.vector_repository import VectorRepository
from src.services.embedding_service import EmbeddingService
from src.vectorstore.collection_manager import ensure_collection
from src.vectorstore.qdrant_client import get_qdrant_client

logger = logging.getLogger("second_brain.ingestion")


class IngestionService:
    def __init__(
        self,
        document_repository: DocumentRepository,
        vector_repository: VectorRepository,
        embedding_service: EmbeddingService,
        text_splitter: TextSplitterService | None = None,
    ):
        self.document_repository = document_repository
        self.vector_repository = vector_repository
        self.embedding_service = embedding_service
        self.text_splitter = text_splitter or TextSplitterService()

    def process_document(self, document: Document) -> None:
        """
        Runs the full pipeline for a single document and persists the
        resulting status (COMPLETED + chunk_count, or FAILED +
        error_message) back to Postgres. Never raises — failures are
        captured on the Document row so the user can see what went wrong
        via GET /api/documents/{id}/status.
        """
        self.document_repository.update_status(document, ProcessingStatus.PROCESSING)

        try:
            loader = get_loader(document.file_type)
            pages = loader.load(Path(document.file_path))

            chunks = self.text_splitter.split_pages(pages)
            if not chunks:
                raise DocumentProcessingException(
                    "Document produced no usable text chunks after cleaning/splitting."
                )

            embeddings = self.embedding_service.embed_documents([c.content for c in chunks])

            written = self.vector_repository.upsert_chunks(
                user_id=document.user_id,
                document_id=document.id,
                filename=document.filename,
                chunks=chunks,
                embeddings=embeddings,
            )

            self.document_repository.update_status(
                document, ProcessingStatus.COMPLETED, chunk_count=written
            )
            logger.info(
                "Document %s processed successfully: %d chunks", document.id, written
            )

        except Exception as exc:  # noqa: BLE001 - intentionally broad: this is a terminal error boundary
            logger.exception("Failed to process document %s", document.id)
            self.document_repository.update_status(
                document,
                ProcessingStatus.FAILED,
                chunk_count=0,
                error_message=str(exc)[:2000],
            )


def build_ingestion_service(db: Session) -> IngestionService:
    """
    Convenience factory for use from background tasks / workers, where
    FastAPI's `Depends` injection isn't available and dependencies must
    be constructed manually.
    """
    client = get_qdrant_client()
    ensure_collection(client)

    return IngestionService(
        document_repository=DocumentRepository(db),
        vector_repository=VectorRepository(client),
        embedding_service=EmbeddingService(),
    )


def process_document_task(document_id: UUID) -> None:
    """
    Entry point suitable for a BackgroundTask or a Celery/RQ task body.

    Deliberately opens its OWN database session rather than reusing the
    session from the HTTP request that triggered it: FastAPI's `get_db`
    dependency closes its session as soon as the request/response cycle
    finishes, which happens BEFORE a `BackgroundTasks` callback runs (and
    a Celery worker wouldn't have access to that session at all). Reusing
    a closed session here would fail, so this function is fully
    self-contained — the same shape it would need if run inside a
    Celery/RQ task, so promoting it later is a drop-in change.

    The document is looked up without a user_id filter here deliberately:
    this runs as an internal system task after the upload endpoint has
    already verified ownership, not in response to a new user-supplied
    document_id from an inbound request.
    """
    db = SessionLocal()
    try:
        document = db.get(Document, document_id)
        if document is None:
            logger.warning("process_document_task: document %s no longer exists", document_id)
            return

        service = build_ingestion_service(db)
        service.process_document(document)
    finally:
        db.close()
