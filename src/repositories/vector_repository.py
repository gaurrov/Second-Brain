"""
Vector repository — the ONLY module that talks to Qdrant for chunk
storage/retrieval/deletion.

CRITICAL ISOLATION RULE: `upsert_chunks` always writes `user_id` into
every point's payload, and `delete_by_document` always filters by BOTH
`document_id` AND `user_id`. There is intentionally no method that
deletes or searches by `document_id` alone — see the docstrings below.
Query/search-by-similarity (used by the RAG retrieval flow) is added in
the chat/retrieval module; this module currently covers the write and
delete paths needed for document ingestion and management.
"""
import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.core.config import settings
from src.rag.splitters.text_splitter import TextChunk

logger = logging.getLogger("second_brain.vector_repository")


class VectorRepository:
    def __init__(self, client: QdrantClient):
        self.client = client
        self.collection_name = settings.QDRANT_COLLECTION_NAME

    def upsert_chunks(
        self,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
        filename: str,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> int:
        """
        Upsert one Qdrant point per chunk. Every point's payload carries
        {user_id, document_id, filename, chunk_index, page_number,
        content} — user_id and document_id are what make the later
        RAG-retrieval filter (`user_id == current_user.id`) possible and
        mandatory.

        Returns the number of points written.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")

        points = [
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "user_id": str(user_id),
                    "document_id": str(document_id),
                    "filename": filename,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "content": chunk.content,
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.debug(
                "Upserted %d vectors for document=%s user=%s",
                len(points), document_id, user_id,
            )

        return len(points)

    def delete_by_document(self, document_id: uuid.UUID, user_id: uuid.UUID) -> None:
        """
        Delete every vector belonging to a document. Filtered by BOTH
        document_id and user_id so that even a caller who somehow got
        hold of another user's document_id cannot delete those vectors
        without also matching that user's own user_id.
        """
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
        logger.debug(
            "Deleted vectors for document=%s user=%s", document_id, user_id
        )

    def count_by_document(self, document_id: uuid.UUID, user_id: uuid.UUID) -> int:
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
        return result.count
