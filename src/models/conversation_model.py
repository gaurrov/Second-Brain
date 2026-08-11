"""
Conversation ORM model.

Represents a single chat thread belonging to exactly one user. This is the
relational "system of record" for RAG conversations: every message written
by the RAG pipeline is attached to a conversation, and the conversation
carries the `user_id` that all downstream ownership checks filter on.
"""
import uuid

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from src.models.user_model import User


class Conversation(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "conversations"

    # Ownership — the cornerstone of per-user isolation for this table.
    # Every repository query against `conversations` must filter on this.
    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(String(255), nullable=False)

    user: Mapped[User] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Conversation id={self.id} title={self.title!r}>"
