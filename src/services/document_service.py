"""
Document service — the use-case layer for document management.

Handles the upload flow (validate -> store file -> create DB row with
status PENDING). Processing (extract/chunk/embed/store vectors) is
dispatched via a `dispatch_processing` callable injected at construction
time — this keeps the service decoupled from FastAPI's BackgroundTasks
and makes swapping to Celery/RQ a one-line change. Every method that
touches an existing document requires `user_id` and delegates to
`DocumentRepository.get_by_id_for_user`, so ownership is enforced before
any read, status check, or delete can proceed.
"""
import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from sqlalchemy.orm import Session

from src.core.exceptions import DocumentNotFoundException
from src.models.document_model import Document
from src.repositories.document_repository import DocumentRepository
from src.repositories.vector_repository import VectorRepository
from src.utils.file_utils import build_storage_path, delete_document_directory, save_upload_file, validate_file_type, validate_magic_bytes
from src.vectorstore.qdrant_client import get_qdrant_client

logger = logging.getLogger("second_brain.document_service")


class DocumentService:
    def __init__(
        self,
        document_repository: DocumentRepository,
        db: Session,
        dispatch_processing: Callable[[uuid.UUID], Any] | None = None,
    ):
        self.document_repository = document_repository
        self.db = db
        self._dispatch_processing = dispatch_processing

    async def upload(self, file: UploadFile, user_id: uuid.UUID) -> Document:
        """
        Validate -> store on disk -> create Document row (PENDING).
        Returns immediately after the DB row is created; processing is
        dispatched via the injected callback if one was provided.
        """
        file_type = validate_file_type(file.filename, file.content_type)

        document_id = uuid.uuid4()
        storage_path = build_storage_path(user_id, document_id, file.filename)
        bytes_written = await save_upload_file(file, storage_path)

        validate_magic_bytes(storage_path, file_type)

        document = Document(
            id=document_id,
            user_id=user_id,
            filename=Path(file.filename).name,
            file_type=file_type,
            file_path=str(storage_path),
            file_size_bytes=bytes_written,
        )
        document = self.document_repository.create(document)

        if self._dispatch_processing is not None:
            self._dispatch_processing(document.id)
        else:
            logger.warning(
                "Document %s created with no dispatch_processing callback configured; "
                "it will remain stuck in PENDING until processed manually.",
                document.id,
            )

        return document

    def get_document(self, document_id: uuid.UUID, user_id: uuid.UUID) -> Document:
        document = self.document_repository.get_by_id_for_user(document_id, user_id)
        if document is None:
            raise DocumentNotFoundException()
        return document

    def list_documents(
        self, user_id: uuid.UUID, limit: int = 50, offset: int = 0
    ) -> tuple[list[Document], int]:
        return self.document_repository.list_for_user(user_id, limit=limit, offset=offset)

    def delete_document(self, document_id: uuid.UUID, user_id: uuid.UUID) -> None:
        """
        Deletes the document's DB row, its vectors in Qdrant, and its
        file(s) on disk. All three deletions are scoped to (document_id,
        user_id) so a user can only ever delete their own data.
        """
        document = self.get_document(document_id, user_id)  # raises if not owned by this user

        vector_repository = VectorRepository(get_qdrant_client())
        vector_repository.delete_by_document(document_id=document.id, user_id=user_id)

        delete_document_directory(user_id=user_id, document_id=document.id)

        deleted = self.document_repository.delete_for_user(document_id, user_id)
        if not deleted:
            # Should be unreachable: get_document() above already confirmed
            # ownership. Logged defensively in case of a concurrent delete
            # racing this request between the two calls.
            logger.warning(
                "delete_for_user found no row for document=%s user=%s after get_document succeeded",
                document_id, user_id,
            )
