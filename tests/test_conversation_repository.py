"""
Unit tests for ConversationRepository and MessageRepository isolation.

Verifies that every read/delete path is ownership-scoped by user_id and
that unscoped `get_by_id` is structurally disabled.
"""
import uuid

import pytest

from src.core.constants import MessageRole
from src.models.conversation_model import Conversation
from src.models.message_model import Message
from src.repositories.conversation_repository import ConversationRepository
from src.repositories.message_repository import MessageRepository


@pytest.fixture
def conversation_repo(db_session):
    return ConversationRepository(db_session)


@pytest.fixture
def message_repo(db_session):
    return MessageRepository(db_session)


def _create_conversation(repo, user_id, title="A conversation"):
    return repo.create(Conversation(user_id=user_id, title=title))


def _create_message(repo, conversation_id, user_id, role, content):
    return repo.create(
        Message(conversation_id=conversation_id, user_id=user_id, role=role, content=content)
    )


class TestConversationRepository:
    def test_get_by_id_disabled(self, conversation_repo):
        with pytest.raises(NotImplementedError):
            conversation_repo.get_by_id(uuid.uuid4())

    def test_create_and_get_by_id_for_user(self, conversation_repo):
        user_id = uuid.uuid4()
        conversation = _create_conversation(conversation_repo, user_id)
        fetched = conversation_repo.get_by_id_for_user(conversation.id, user_id)
        assert fetched is not None
        assert fetched.id == conversation.id
        assert fetched.user_id == user_id

    def test_cross_user_get_returns_none(self, conversation_repo):
        owner = uuid.uuid4()
        stranger = uuid.uuid4()
        conversation = _create_conversation(conversation_repo, owner)
        assert conversation_repo.get_by_id_for_user(conversation.id, stranger) is None

    def test_list_for_user_is_scoped(self, conversation_repo):
        user_a, user_b = uuid.uuid4(), uuid.uuid4()
        _create_conversation(conversation_repo, user_a, title="A one")
        _create_conversation(conversation_repo, user_a, title="A two")
        _create_conversation(conversation_repo, user_b, title="B one")

        list_a, total_a = conversation_repo.list_for_user(user_a)
        list_b, total_b = conversation_repo.list_for_user(user_b)

        assert total_a == 2
        assert {c.title for c in list_a} == {"A one", "A two"}
        assert total_b == 1
        assert list_b[0].title == "B one"

    def test_delete_for_user_only_own(self, conversation_repo):
        owner, stranger = uuid.uuid4(), uuid.uuid4()
        conversation = _create_conversation(conversation_repo, owner)

        assert conversation_repo.delete_for_user(conversation.id, stranger) is False
        assert conversation_repo.get_by_id_for_user(conversation.id, owner) is not None

        assert conversation_repo.delete_for_user(conversation.id, owner) is True
        assert conversation_repo.get_by_id_for_user(conversation.id, owner) is None


class TestMessageRepository:
    def test_list_for_conversation_is_scoped_to_user(self, db_session, conversation_repo, message_repo):
        user_a, user_b = uuid.uuid4(), uuid.uuid4()
        conv_a = _create_conversation(conversation_repo, user_a, title="A")
        conv_b = _create_conversation(conversation_repo, user_b, title="B")

        _create_message(message_repo, conv_a.id, user_a, MessageRole.USER, "A question")
        _create_message(message_repo, conv_a.id, user_a, MessageRole.ASSISTANT, "A answer")
        _create_message(message_repo, conv_b.id, user_b, MessageRole.USER, "B question")

        # User A sees only their own messages in their conversation.
        messages = message_repo.list_for_conversation(conv_a.id, user_a, limit=10)
        assert [m.content for m in messages] == ["A question", "A answer"]

        # User B asking for conv_a's messages gets nothing (ownership filter).
        assert message_repo.list_for_conversation(conv_a.id, user_b, limit=10) == []

        # User B sees their own conversation's messages normally.
        assert [m.content for m in message_repo.list_for_conversation(conv_b.id, user_b, limit=10)] == [
            "B question"
        ]

    def test_list_returns_most_recent_first_in_chronological_order(
        self, conversation_repo, message_repo
    ):
        user_id = uuid.uuid4()
        conversation = _create_conversation(conversation_repo, user_id)
        contents = [f"message-{i}" for i in range(5)]
        for content in contents:
            _create_message(message_repo, conversation.id, user_id, MessageRole.USER, content)

        # limit=3 returns the LAST 3, oldest-first among them.
        messages = message_repo.list_for_conversation(conversation.id, user_id, limit=3)
        assert [m.content for m in messages] == ["message-2", "message-3", "message-4"]

        assert message_repo.count_for_conversation(conversation.id, user_id) == 5

    def test_count_is_scoped_to_user(self, conversation_repo, message_repo):
        user_a, user_b = uuid.uuid4(), uuid.uuid4()
        shared_conv = _create_conversation(conversation_repo, user_a, title="shared")
        _create_message(message_repo, shared_conv.id, user_a, MessageRole.USER, "A msg")

        assert message_repo.count_for_conversation(shared_conv.id, user_a) == 1
        assert message_repo.count_for_conversation(shared_conv.id, user_b) == 0

    def test_deleting_conversation_cascades_to_messages(
        self, conversation_repo, message_repo
    ):
        user_id = uuid.uuid4()
        conversation = _create_conversation(conversation_repo, user_id)
        _create_message(message_repo, conversation.id, user_id, MessageRole.USER, "msg")

        assert conversation_repo.delete_for_user(conversation.id, user_id) is True
        assert message_repo.count_for_conversation(conversation.id, user_id) == 0
