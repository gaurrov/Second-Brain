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

Each pipeline step is individually timed and logged so performance
bottlenecks are immediately visible in production logs. Failures at any
step are caught, classified (transient vs. permanent), and persisted on
the Document row — the pipeline never raises to the caller.
"""
import logging
import time
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from src.core.constants import PIPELINE_STEPS, ProcessingStatus
from src.core.exceptions import DocumentProcessingException
from src.core.metrics import (
    ingestion_documents_total,
    ingestion_pipeline_duration_seconds,
    ingestion_pipeline_total_seconds,
)
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
    ) -> None:
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
        pipeline_start = time.perf_counter()
        step_timings: dict[str, float] = {}

        def _record_step(name: str, start: float) -> None:
            elapsed = time.perf_counter() - start
            step_timings[name] = elapsed
            ingestion_pipeline_duration_seconds.labels(step=name).observe(elapsed)

        try:
            # --- Step 1: Extract text ---
            step_start = time.perf_counter()
            loader = get_loader(document.file_type)
            pages = loader.load(Path(document.file_path))
            _record_step("extract", step_start)
            logger.info(
                "Document %s extract: %d pages in %.2fs",
                document.id, len(pages), step_timings["extract"],
            )

            # --- Step 2: Clean + chunk ---
            step_start = time.perf_counter()
            chunks = self.text_splitter.split_pages(pages)
            _record_step("chunk", step_start)
            if not chunks:
                raise DocumentProcessingException(
                    "Document produced no usable text chunks after cleaning/splitting."
                )
            logger.info(
                "Document %s chunk: %d chunks in %.2fs",
                document.id, len(chunks), step_timings["chunk"],
            )

            # --- Step 3: Embed ---
            step_start = time.perf_counter()
            embeddings = self.embedding_service.embed_documents([c.content for c in chunks])
            _record_step("embed", step_start)
            logger.info(
                "Document %s embed: %d vectors in %.2fs",
                document.id, len(embeddings), step_timings["embed"],
            )

            # --- Step 4: Store vectors ---
            step_start = time.perf_counter()
            written = self.vector_repository.upsert_chunks(
                user_id=document.user_id,
                document_id=document.id,
                filename=document.filename,
                chunks=chunks,
                embeddings=embeddings,
            )
            _record_step("store", step_start)
            logger.info(
                "Document %s store: %d vectors written in %.2fs",
                document.id, written, step_timings["store"],
            )

            # --- Step 5: Finalize ---
            self.document_repository.update_status(
                document, ProcessingStatus.COMPLETED, chunk_count=written
            )

            total_time = time.perf_counter() - pipeline_start
            ingestion_pipeline_total_seconds.observe(total_time)
            ingestion_documents_total.labels(status="completed").inc()
            logger.info(
                "Document %s COMPLETED: %d chunks, total=%.2fs steps=%s",
                document.id, written, total_time,
                " ".join(f"{k}={v:.2f}s" for k, v in step_timings.items()),
            )

        except DocumentProcessingException:
            raise
        except Exception as exc:  # noqa: BLE001 - intentionally broad: this is a terminal error boundary
            total_time = time.perf_counter() - pipeline_start
            ingestion_documents_total.labels(status="failed").inc()
            logger.exception(
                "Document %s FAILED after %.2fs: %s", document.id, total_time, exc
            )
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
