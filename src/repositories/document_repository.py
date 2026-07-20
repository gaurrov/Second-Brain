"""
Document repository.

CRITICAL ISOLATION RULE: every read/update/delete method here takes
`user_id` as a mandatory parameter and includes it in the WHERE clause.
There is deliberately no `get_by_id(document_id)` method without a
user_id — that shape of method would make it possible for a future
caller to forget the ownership filter. See `get_by_id_for_user`.
"""
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.core.constants import ProcessingStatus
from src.models.document_model import Document
from src.repositories.base_repository import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    def __init__(self, db: Session):
        super().__init__(model=Document, db=db)

    def get_by_id(self, record_id: uuid.UUID) -> Document | None:
        """
        Intentionally disabled. `BaseRepository.get_by_id` has no
        ownership filter, and Document is a user-owned resource — calling
        this directly would bypass multi-tenant isolation. Use
        `get_by_id_for_user` instead.
        """
        raise NotImplementedError(
            "DocumentRepository.get_by_id() is disabled to prevent unscoped access. "
            "Use get_by_id_for_user(document_id, user_id) instead."
        )

    def get_by_id_for_user(self, document_id: uuid.UUID, user_id: uuid.UUID) -> Document | None:
        """The only way to fetch a single document — always ownership-scoped."""
        stmt = select(Document).where(
            Document.id == document_id,
            Document.user_id == user_id,
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def list_for_user(
        self, user_id: uuid.UUID, limit: int = 50, offset: int = 0
    ) -> tuple[list[Document], int]:
        filters = (Document.user_id == user_id,)

        count_stmt = select(func.count()).select_from(Document).where(*filters)
        total = self.db.execute(count_stmt).scalar_one()

        stmt = (
            select(Document)
            .where(*filters)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        documents = list(self.db.execute(stmt).scalars().all())

        return documents, total

    def delete_for_user(self, document_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """
        Ownership-scoped delete. Returns True if a row was removed, False if
        the document does not exist for this user.
        """
        document = self.get_by_id_for_user(document_id, user_id)
        if document is None:
            return False
        self.delete(document)
        return True

    def update_status(
        self,
        document: Document,
        status: ProcessingStatus,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> Document:
        document.processing_status = status
        if chunk_count is not None:
            document.chunk_count = chunk_count
        document.error_message = error_message
        self.db.commit()
        self.db.refresh(document)
        return document
