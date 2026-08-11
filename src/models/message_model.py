"""
Message ORM model.

A single turn (user question or assistant answer) inside a conversation.

Carries its OWN `user_id` in addition to `conversation_id` so that every
read of messages can be filtered by BOTH columns. This makes cross-user
access structurally impossible even if a caller somehow obtains another
user's conversation_id: a message query that filters `conversation_id`
without also matching the caller's `user_id` returns nothing.

`created_at` is given a Python-side default (microsecond precision) in
addition to the server default so that the message ordering used for chat
history is correct even on SQLite, whose CURRENT_TIMESTAMP only has
second precision and would otherwise give every rapid insert the same
timestamp.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime
from sqlalchemy import Enum as SAEnum
from sqlalchemy import ForeignKey, Index, Text, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.constants import MessageRole
from src.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from src.models.user_model import User


class Message(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_user_conversation", "user_id", "conversation_id"),
    )

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[MessageRole] = mapped_column(
        SAEnum(
            MessageRole,
            native_enum=False,
            length=10,
            validate_strings=True,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
    user: Mapped[User] = relationship()

    def __repr__(self) -> str:
        return f"<Message id={self.id} role={self.role!r}>"
