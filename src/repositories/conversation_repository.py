"""
Conversation repository.

CRITICAL ISOLATION RULE: every read/update/delete method here takes
`user_id` as a mandatory parameter and includes it in the WHERE clause.
There is deliberately no `get_by_id(conversation_id)` method without a
user_id — that shape of method would make it possible for a future
caller to forget the ownership filter. See `get_by_id_for_user`.
"""
import logging
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.models.conversation_model import Conversation
from src.repositories.base_repository import BaseRepository

logger = logging.getLogger("second_brain.conversation_repository")


class ConversationRepository(BaseRepository[Conversation]):
    def __init__(self, db: Session) -> None:
        super().__init__(model=Conversation, db=db)

    def get_by_id(self, record_id: uuid.UUID) -> Conversation | None:
        """
        Intentionally disabled. `BaseRepository.get_by_id` has no
        ownership filter, and Conversation is a user-owned resource —
        calling this directly would bypass multi-tenant isolation. Use
        `get_by_id_for_user` instead.
        """
        raise NotImplementedError(
            "ConversationRepository.get_by_id() is disabled to prevent unscoped access. "
            "Use get_by_id_for_user(conversation_id, user_id) instead."
        )

    def get_by_id_for_user(
        self, conversation_id: uuid.UUID, user_id: uuid.UUID
    ) -> Conversation | None:
        """The only way to fetch a single conversation — always ownership-scoped."""
        stmt = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def list_for_user(
        self, user_id: uuid.UUID, limit: int = 50, offset: int = 0
    ) -> tuple[list[Conversation], int]:
        filters = (Conversation.user_id == user_id,)

        count_stmt = select(func.count()).select_from(Conversation).where(*filters)
        total = self.db.execute(count_stmt).scalar_one()

        stmt = (
            select(Conversation)
            .where(*filters)
            .order_by(Conversation.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        conversations = list(self.db.execute(stmt).scalars().all())

        return conversations, total

    def delete_for_user(self, conversation_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """
        Ownership-scoped delete (messages cascade via the ORM relationship
        and the FK `ondelete="CASCADE"`). Returns True if a row was
        removed, False if the conversation does not exist for this user.
        """
        conversation = self.get_by_id_for_user(conversation_id, user_id)
        if conversation is None:
            return False
        self.delete(conversation)
        return True
