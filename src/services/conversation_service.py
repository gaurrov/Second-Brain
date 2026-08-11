"""
Conversation service - API-facing CRUD for conversations and messages.

Ownership is enforced on every method: the caller's `user_id` is always
required and every repository call is user-scoped, so a user can never
read or delete another user's conversation even if they know its id.
"""
import logging
import uuid

from src.core.exceptions import ConversationNotFoundException
from src.models.conversation_model import Conversation
from src.models.message_model import Message
from src.repositories.conversation_repository import ConversationRepository
from src.repositories.message_repository import MessageRepository

logger = logging.getLogger("second_brain.conversation_service")


class ConversationService:
    def __init__(
        self,
        conversation_repository: ConversationRepository,
        message_repository: MessageRepository,
    ) -> None:
        self.conversation_repository = conversation_repository
        self.message_repository = message_repository

    def get_conversation(self, conversation_id: uuid.UUID, user_id: uuid.UUID) -> Conversation:
        conversation = self.conversation_repository.get_by_id_for_user(conversation_id, user_id)
        if conversation is None:
            raise ConversationNotFoundException()
        return conversation

    def get_conversation_detail(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[Conversation, list[Message]]:
        """Fetch a conversation with its messages, ownership-checked."""
        conversation = self.get_conversation(conversation_id, user_id)
        messages = self.message_repository.list_for_conversation(
            conversation_id, user_id, limit=limit, offset=offset
        )
        return conversation, messages

    def list_conversations(
        self, user_id: uuid.UUID, limit: int = 50, offset: int = 0
    ) -> tuple[list[Conversation], int]:
        return self.conversation_repository.list_for_user(user_id, limit=limit, offset=offset)

    def list_messages(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Message]:
        # Ownership gate: raises 404 unless this user owns the conversation.
        self.get_conversation(conversation_id, user_id)
        return self.message_repository.list_for_conversation(
            conversation_id, user_id, limit=limit, offset=offset
        )

    def delete_conversation(self, conversation_id: uuid.UUID, user_id: uuid.UUID) -> None:
        deleted = self.conversation_repository.delete_for_user(conversation_id, user_id)
        if not deleted:
            raise ConversationNotFoundException()
