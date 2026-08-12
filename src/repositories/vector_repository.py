"""
Vector repository — the ONLY module that talks to Qdrant for chunk
storage/retrieval/deletion.

CRITICAL ISOLATION RULE: `upsert_chunks` always writes `user_id` into
every point's payload, and every delete/search path always filters by
`user_id`. There is intentionally no method that deletes or searches by
`document_id` alone — see the docstrings below.

Every payload carries:
    {user_id, document_id, filename, page_number, chunk_index,
     content, timestamp}

where `timestamp` is the ISO-8601 UTC ingestion time. `user_id` and
`document_id` (plus the other filterable fields) have payload indexes
provisioned by `src.vectorstore.collection_manager`.
"""
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.core.config import settings
from src.core.metrics import (
    vector_operation_duration_seconds,
    vector_operations_total,
    vector_points_upserted_total,
)
from src.rag.splitters.text_splitter import TextChunk
from src.utils.retry import retry

logger = logging.getLogger("second_brain.vector_repository")


def _is_retryable_qdrant_error(exc: BaseException) -> bool:
    """Transient Qdrant failures (transport + 5xx) are retried; 4xx are not."""
    if isinstance(exc, (TimeoutError, ConnectionError, OSError, BrokenPipeError)):
        return True
    try:
        from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

        if isinstance(exc, ResponseHandlingException):
            return True
        if isinstance(exc, UnexpectedResponse):
            code = getattr(exc, "status_code", None)
            return code is None or code >= 500
    except ImportError:
        pass
    return False


@dataclass(frozen=True)
class SearchResult:
    """A single point returned by a similarity search, fully typed."""

    point_id: str
    score: float
    user_id: str
    document_id: str
    filename: str
    page_number: int | None
    chunk_index: int
    content: str
    timestamp: str | None


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string, e.g. 2026-07-31T12:00:00+00:00."""
    return datetime.now(timezone.utc).isoformat()


class VectorRepository:
    def __init__(self, client: QdrantClient, collection_name: str | None = None) -> None:
        self.client = client
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME

    @retry(target="qdrant", retry_on=_is_retryable_qdrant_error)
    def upsert_chunks(
        self,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
        filename: str,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
        *,
        timestamp: str | None = None,
    ) -> int:
        """
        Upsert one Qdrant point per chunk. Every point's payload carries
        {user_id, document_id, filename, chunk_index, page_number,
        content, timestamp} — user_id and document_id are what make the
        later retrieval filter (`user_id == current_user.id`) possible
        and mandatory.

        Points are uploaded in batches of QDRANT_UPSERT_BATCH_SIZE using
        Qdrant's columnar `Batch` form (parallel ids/vectors/payloads
        lists) rather than a list of PointStruct objects, which avoids
        materializing hundreds of intermediate objects for large
        documents and keeps peak memory flat.

        Returns the number of points written.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        if not chunks:
            return 0

        stored_at = timestamp or _utc_now_iso()
        point_ids = [str(uuid.uuid4()) for _ in chunks]
        payloads = [
            {
                "user_id": str(user_id),
                "document_id": str(document_id),
                "filename": filename,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "timestamp": stored_at,
            }
            for chunk in chunks
        ]

        batch_size = max(settings.QDRANT_UPSERT_BATCH_SIZE, 1)
        written = 0
        try:
            with vector_operation_duration_seconds.labels(operation="upsert").time():
                for start in range(0, len(chunks), batch_size):
                    end = min(start + batch_size, len(chunks))
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=qmodels.Batch(
                            ids=point_ids[start:end],
                            vectors=embeddings[start:end],
                            payloads=payloads[start:end],
                        ),
                    )
                    written += end - start
        except Exception:
            vector_operations_total.labels(operation="upsert", outcome="error").inc()
            raise
        vector_operations_total.labels(operation="upsert", outcome="success").inc()
        vector_points_upserted_total.labels(operation="upsert").inc(written)

        logger.debug(
            "Upserted %d vectors for document=%s user=%s",
            written, document_id, user_id,
        )
        return written

    @retry(target="qdrant", retry_on=_is_retryable_qdrant_error)
    def search(
        self,
        query_vector: list[float],
        user_id: uuid.UUID,
        *,
        document_id: uuid.UUID | None = None,
        page_number: int | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """
        Cosine-similarity search over the caller's own chunks.

        The `user_id` filter is always applied; `document_id` /
        `page_number` are optional narrowing filters. There is no search
        that omits the user_id filter — cross-user retrieval is
        structurally impossible through this method.
        """
        conditions: list[qmodels.FieldCondition] = [
            qmodels.FieldCondition(
                key="user_id", match=qmodels.MatchValue(value=str(user_id))
            )
        ]
        if document_id is not None:
            conditions.append(
                qmodels.FieldCondition(
                    key="document_id", match=qmodels.MatchValue(value=str(document_id))
                )
            )
        if page_number is not None:
            conditions.append(
                qmodels.FieldCondition(
                    key="page_number", match=qmodels.MatchValue(value=page_number)
                )
            )

        try:
            with vector_operation_duration_seconds.labels(operation="search").time():
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=qmodels.Filter(must=conditions),
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
        except Exception:
            vector_operations_total.labels(operation="search", outcome="error").inc()
            raise
        vector_operations_total.labels(operation="search", outcome="success").inc()
        return [self._to_search_result(point) for point in response.points]

    @retry(target="qdrant", retry_on=_is_retryable_qdrant_error)
    def delete_by_document(self, document_id: uuid.UUID, user_id: uuid.UUID) -> None:
        """
        Delete every vector belonging to a document. Filtered by BOTH
        document_id and user_id so that even a caller who somehow got
        hold of another user's document_id cannot delete those vectors
        without also matching that user's own user_id.
        """
        try:
            with vector_operation_duration_seconds.labels(operation="delete").time():
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=qmodels.FilterSelector(
                        filter=qmodels.Filter(
                            must=[
                                qmodels.FieldCondition(
                                    key="document_id", match=qmodels.MatchValue(value=str(document_id))
                                ),
                                qmodels.FieldCondition(
                                    key="user_id", match=qmodels.MatchValue(value=str(user_id))
                                ),
                            ]
                        )
                    ),
                )
        except Exception:
            vector_operations_total.labels(operation="delete", outcome="error").inc()
            raise
        vector_operations_total.labels(operation="delete", outcome="success").inc()
        logger.debug(
            "Deleted vectors for document=%s user=%s", document_id, user_id
        )

    @retry(target="qdrant", retry_on=_is_retryable_qdrant_error)
    def count_by_document(self, document_id: uuid.UUID, user_id: uuid.UUID) -> int:
        """Number of stored chunks for a document (scoped by user)."""
        try:
            with vector_operation_duration_seconds.labels(operation="count").time():
                result = self.client.count(
                    collection_name=self.collection_name,
                    count_filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_id", match=qmodels.MatchValue(value=str(document_id))
                            ),
                            qmodels.FieldCondition(
                                key="user_id", match=qmodels.MatchValue(value=str(user_id))
                            ),
                        ]
                    ),
                )
        except Exception:
            vector_operations_total.labels(operation="count", outcome="error").inc()
            raise
        vector_operations_total.labels(operation="count", outcome="success").inc()
        return result.count

    @staticmethod
    def _to_search_result(point) -> SearchResult:
        payload = point.payload or {}
        return SearchResult(
            point_id=str(point.id),
            score=float(point.score),
            user_id=str(payload.get("user_id", "")),
            document_id=str(payload.get("document_id", "")),
            filename=payload.get("filename", ""),
            page_number=payload.get("page_number"),
            chunk_index=int(payload.get("chunk_index", -1)),
            content=payload.get("content", ""),
            timestamp=payload.get("timestamp"),
        )
