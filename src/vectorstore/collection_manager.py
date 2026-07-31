"""
Qdrant collection lifecycle management.

Responsible for making sure the configured collection exists with the
right vector configuration, and that every payload field that filters
searches/deletes has a payload index. The `user_id` index in particular
is the multi-tenant isolation mechanism — every Qdrant query must filter
on `user_id`, and an index is what keeps those filtered scans fast at
scale instead of doing a full collection scan.
"""
import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.core.config import settings

logger = logging.getLogger("second_brain.vectorstore")

# Payload fields that carry filtering metadata, and the index schema each
# requires. `content` and `timestamp` are stored in every payload but not
# indexed — there is no exact-match/range query on them today, and
# indexing large text fields wastes memory for no benefit.
INDEXED_PAYLOAD_FIELDS: dict[str, qmodels.PayloadSchemaType] = {
    "user_id": qmodels.PayloadSchemaType.KEYWORD,
    "document_id": qmodels.PayloadSchemaType.KEYWORD,
    "filename": qmodels.PayloadSchemaType.KEYWORD,
    "page_number": qmodels.PayloadSchemaType.INTEGER,
    "chunk_index": qmodels.PayloadSchemaType.INTEGER,
}


def ensure_collection(client: QdrantClient) -> None:
    """
    Idempotently ensures the configured collection exists with the
    correct vector size/distance, and that payload indexes exist for
    every indexed field. Safe to call on every app startup.
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

    for field_name, schema in INDEXED_PAYLOAD_FIELDS.items():
        _ensure_payload_index(client, collection_name, field_name, schema)


def _ensure_payload_index(
    client: QdrantClient,
    collection_name: str,
    field_name: str,
    schema: qmodels.PayloadSchemaType,
) -> None:
    """
    Create a payload index, tolerating a pre-existing index on the same
    field. Qdrant no-ops an identical index; if a conflicting index type
    already exists (e.g. a dev collection built before a schema change),
    log and continue rather than failing app startup.
    """
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema,
        )
    except Exception as exc:  # noqa: BLE001 - collection provisioning must not brick startup
        logger.warning(
            "Payload index '%s' could not be created on '%s': %s",
            field_name, collection_name, exc,
        )
