"""
Message repository.

Every read here is scoped by BOTH `conversation_id` and `user_id`. Since a
Message row carries its own `user_id`, no query can ever leak one user's
messages to another even if a foreign conversation_id is supplied.
"""
import logging
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.models.message_model import Message
from src.repositories.base_repository import BaseRepository

logger = logging.getLogger("second_brain.message_repository")


class MessageRepository(BaseRepository[Message]):
    def __init__(self, db: Session) -> None:
        super().__init__(model=Message, db=db)

    def list_for_conversation(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Message]:
        """
        Return the most recent `limit` messages of a conversation in
        chronological order. Filtered by BOTH conversation_id and user_id
        so ownership is enforced on every read.
        """
        stmt = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.user_id == user_id,
            )
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
            .offset(offset)
        )
        # Descending fetch + reverse -> the LAST `limit` messages, oldest first.
        return list(reversed(list(self.db.execute(stmt).scalars().all())))

    def count_for_conversation(self, conversation_id: uuid.UUID, user_id: uuid.UUID) -> int:
        stmt = select(func.count()).select_from(Message).where(
            Message.conversation_id == conversation_id,
            Message.user_id == user_id,
        )
        return self.db.execute(stmt).scalar_one()
