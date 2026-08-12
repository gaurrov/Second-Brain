"""
Document ORM model.

Represents a single uploaded file belonging to exactly one user. This is
the relational "system of record" for a document's lifecycle; the actual
extracted text chunks and their embeddings live in Qdrant (see
src/vectorstore/), keyed back to this row via `document_id` + `user_id`
in the Qdrant point payload.
"""
import uuid

from sqlalchemy import Enum as SAEnum
from sqlalchemy import ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.constants import FileType, ProcessingStatus
from src.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from src.models.user_model import User


class Document(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "documents"
    __table_args__ = (
        # Speeds up list_for_user (WHERE user_id=? ORDER BY created_at DESC).
        Index("ix_documents_user_id_created_at", "user_id", "created_at"),
    )

    # Ownership — the cornerstone of per-user isolation for this table.
    # Every repository query against `documents` must filter on this.
    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[FileType] = mapped_column(
        SAEnum(
            FileType,
            native_enum=False,
            length=10,
            validate_strings=True,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
    )
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    processing_status: Mapped[ProcessingStatus] = mapped_column(
        SAEnum(
            ProcessingStatus,
            native_enum=False,
            length=20,
            validate_strings=True,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        default=ProcessingStatus.PENDING,
        server_default=ProcessingStatus.PENDING.value,
        index=True,
    )
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped[User] = relationship(back_populates="documents")

    # NOTE: `created_at` from TimestampMixin doubles as `upload_date`;
    # the API schema (see document_schema.py) exposes it under the
    # `upload_date` field name via a validation alias so the column
    # itself stays consistent with the rest of the codebase's
    # TimestampMixin convention.

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename!r} status={self.processing_status}>"
