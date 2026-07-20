"""
Qdrant collection lifecycle management.

Responsible for making sure the shared `documents_kb` collection exists
with the right vector configuration, and that `user_id` / `document_id`
have payload indexes so filtered searches (the isolation mechanism) are
fast at scale rather than doing a full collection scan.
"""
import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.core.config import settings

logger = logging.getLogger("second_brain.vectorstore")


def ensure_collection(client: QdrantClient) -> None:
    """
    Idempotently ensures the configured collection exists with the
    correct vector size/distance, and that keyword payload indexes exist
    on `user_id` and `document_id`. Safe to call on every app startup.
    """
    collection_name = settings.QDRANT_COLLECTION_NAME

    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        logger.info("Creating Qdrant collection '%s'", collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=settings.EMBEDDING_DIMENSION,
                distance=qmodels.Distance.COSINE,
            ),
        )

    # Payload indexes are idempotent to (re)create — Qdrant no-ops if an
    # identical index already exists.
    client.create_payload_index(
        collection_name=collection_name,
        field_name="user_id",
        field_schema=qmodels.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="document_id",
        field_schema=qmodels.PayloadSchemaType.KEYWORD,
    )
