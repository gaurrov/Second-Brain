"""
Integration tests for the chat / conversation HTTP endpoints.

POST /chat is tested against a fake RAGService (recorded calls + canned
results) so the tests stay offline and fast; the conversation endpoints
(GET/DELETE) run against the real ConversationService and repositories to
prove HTTP-level multi-user isolation.
"""
import uuid

import pytest

from src.core.config import settings
from src.core.constants import MessageRole
from src.models.conversation_model import Conversation
from src.models.message_model import Message
from src.models.user_model import User
from src.repositories.conversation_repository import ConversationRepository
from src.repositories.message_repository import MessageRepository
from src.services.rag_service import RAGResult, SourceRef

API_PREFIX = settings.API_V1_PREFIX


def _register_and_login(client, email="jane@example.com", username="jane_doe"):
    client.post(
        f"{API_PREFIX}/auth/register",
        json={"username": username, "email": email, "password": "StrongP@ss123"},
    )
    login_response = client.post(
        f"{API_PREFIX}/auth/login", json={"email": email, "password": "StrongP@ss123"}
    )
    token = login_response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _user_id(db_session, email):
    return db_session.query(User).filter_by(email=email).one().id


def _seed_conversation(db_session, user_id, title="My conversation", message_count=0):
    conv_repo = ConversationRepository(db_session)
    message_repo = MessageRepository(db_session)
    conversation = conv_repo.create(Conversation(user_id=user_id, title=title))
    for i in range(message_count):
        message_repo.create(
            Message(
                conversation_id=conversation.id,
                user_id=user_id,
                role=MessageRole.USER,
                content=f"message-{i}",
            )
        )
    return conversation


class _FakeRAGService:
    def __init__(self, refused=False):
        self.calls = []
        self.refused = refused

    def answer(self, question, user_id, conversation_id=None):
        self.calls.append(
            {
                "question": question,
                "user_id": str(user_id),
                "conversation_id": str(conversation_id) if conversation_id else None,
            }
        )
        return RAGResult(
            answer="I couldn't find enough information." if self.refused else "Test answer.",
            conversation_id=conversation_id or uuid.uuid4(),
            user_message_id=uuid.uuid4(),
            assistant_message_id=uuid.uuid4(),
            refused=self.refused,
            sources=[] if self.refused else [SourceRef(document_id=str(uuid.uuid4()), filename="doc.txt", page_number=1, chunk_index=0, score=0.9, snippet="snippet")],
        )


@pytest.fixture
def fake_rag(client):
    from src.api.deps import get_rag_service

    fake = _FakeRAGService()
    client.app.dependency_overrides[get_rag_service] = lambda: fake
    yield fake
    client.app.dependency_overrides.clear()


class TestChatEndpoint:
    def test_chat_requires_authentication(self, client):
        response = client.post(f"{API_PREFIX}/chat", json={"question": "hello"})
        assert response.status_code == 401

    def test_chat_returns_answer_and_records_caller(self, client, db_session, fake_rag):
        headers = _register_and_login(client)
        user_id = _user_id(db_session, "jane@example.com")

        response = client.post(
            f"{API_PREFIX}/chat", json={"question": "What is my runbook?"}, headers=headers
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Test answer."
        assert body["refused"] is False
        assert len(body["sources"]) == 1
        assert body["sources"][0]["filename"] == "doc.txt"
        assert fake_rag.calls[0]["question"] == "What is my runbook?"
        assert fake_rag.calls[0]["user_id"] == str(user_id)

    def test_chat_refuses_politely_when_context_insufficient(self, client, fake_rag):
        fake_rag.refused = True
        headers = _register_and_login(client)
        response = client.post(
            f"{API_PREFIX}/chat", json={"question": "anything"}, headers=headers
        )
        assert response.status_code == 200
        body = response.json()
        assert body["refused"] is True
        assert body["sources"] == []
        assert "couldn't find enough information" in body["answer"]

    def test_chat_passes_existing_conversation_id(self, client, db_session, fake_rag):
        headers = _register_and_login(client)
        user_id = _user_id(db_session, "jane@example.com")
        conversation = _seed_conversation(db_session, user_id)

        response = client.post(
            f"{API_PREFIX}/chat",
            json={"question": "follow up?", "conversation_id": str(conversation.id)},
            headers=headers,
        )
        assert response.status_code == 200
        assert response.json()["conversation_id"] == str(conversation.id)
        assert fake_rag.calls[0]["conversation_id"] == str(conversation.id)

    def test_chat_rejects_blank_question(self, client, fake_rag):
        headers = _register_and_login(client)
        response = client.post(f"{API_PREFIX}/chat", json={"question": "   "}, headers=headers)
        assert response.status_code == 422

    def test_chat_rejects_overlong_question(self, client, fake_rag):
        headers = _register_and_login(client)
        response = client.post(
            f"{API_PREFIX}/chat",
            json={"question": "a" * (settings.MAX_QUESTION_LENGTH + 1)},
            headers=headers,
        )
        assert response.status_code == 422


class TestConversationEndpoints:
    def test_list_conversations_is_isolated(self, client, db_session):
        headers_a = _register_and_login(client, email="a@example.com", username="user_a")
        headers_b = _register_and_login(client, email="b@example.com", username="user_b")
        user_a = _user_id(db_session, "a@example.com")
        user_b = _user_id(db_session, "b@example.com")

        _seed_conversation(db_session, user_a, title="A conversation")
        _seed_conversation(db_session, user_b, title="B one")
        _seed_conversation(db_session, user_b, title="B two")

        response_a = client.get(f"{API_PREFIX}/conversations", headers=headers_a)
        response_b = client.get(f"{API_PREFIX}/conversations", headers=headers_b)

        assert response_a.json()["total"] == 1
        assert response_a.json()["conversations"][0]["title"] == "A conversation"
        assert response_b.json()["total"] == 2

    def test_get_messages_cross_user_forbidden(self, client, db_session):
        headers_a = _register_and_login(client, email="a@example.com", username="user_a")
        headers_b = _register_and_login(client, email="b@example.com", username="user_b")
        user_a = _user_id(db_session, "a@example.com")
        conversation = _seed_conversation(db_session, user_a, message_count=2)

        response = client.get(
            f"{API_PREFIX}/conversations/{conversation.id}/messages", headers=headers_b
        )
        assert response.status_code == 404

        response_own = client.get(
            f"{API_PREFIX}/conversations/{conversation.id}/messages", headers=headers_a
        )
        assert response_own.status_code == 200
        body = response_own.json()
        assert body["total"] == 2
        assert [m["content"] for m in body["messages"]] == ["message-0", "message-1"]
        assert all(m["role"] == "user" for m in body["messages"])

    def test_get_nonexistent_conversation_messages_404(self, client, db_session):
        headers = _register_and_login(client)
        response = client.get(
            f"{API_PREFIX}/conversations/00000000-0000-0000-0000-000000000000/messages",
            headers=headers,
        )
        assert response.status_code == 404

    def test_delete_cross_user_forbidden(self, client, db_session):
        headers_a = _register_and_login(client, email="a@example.com", username="user_a")
        headers_b = _register_and_login(client, email="b@example.com", username="user_b")
        user_a = _user_id(db_session, "a@example.com")
        conversation = _seed_conversation(db_session, user_a, message_count=1)

        response = client.delete(
            f"{API_PREFIX}/conversations/{conversation.id}", headers=headers_b
        )
        assert response.status_code == 404

        # Still present and readable by its owner.
        assert client.get(
            f"{API_PREFIX}/conversations/{conversation.id}/messages", headers=headers_a
        ).status_code == 200

    def test_delete_own_conversation_succeeds(self, client, db_session):
        headers = _register_and_login(client)
        user_id = _user_id(db_session, "jane@example.com")
        conversation = _seed_conversation(db_session, user_id, message_count=1)

        response = client.delete(
            f"{API_PREFIX}/conversations/{conversation.id}", headers=headers
        )
        assert response.status_code == 204

        assert (
            client.get(
                f"{API_PREFIX}/conversations/{conversation.id}/messages", headers=headers
            ).status_code
            == 404
        )

    def test_get_conversation_detail_includes_messages(self, client, db_session):
        headers = _register_and_login(client)
        user_id = _user_id(db_session, "jane@example.com")
        conversation = _seed_conversation(db_session, user_id, title="Detail", message_count=2)

        response = client.get(f"{API_PREFIX}/conversations/{conversation.id}", headers=headers)

        assert response.status_code == 200
        body = response.json()
        assert body["title"] == "Detail"
        assert len(body["messages"]) == 2
        assert [m["content"] for m in body["messages"]] == ["message-0", "message-1"]

    def test_get_conversation_detail_cross_user_forbidden(self, client, db_session):
        headers_a = _register_and_login(client, email="a@example.com", username="user_a")
        headers_b = _register_and_login(client, email="b@example.com", username="user_b")
        user_a = _user_id(db_session, "a@example.com")
        conversation = _seed_conversation(db_session, user_a, message_count=1)

        response = client.get(f"{API_PREFIX}/conversations/{conversation.id}", headers=headers_b)
        assert response.status_code == 404

        # The owner can still read it.
        assert client.get(
            f"{API_PREFIX}/conversations/{conversation.id}", headers=headers_a
        ).status_code == 200

    def test_get_nonexistent_conversation_detail_404(self, client, db_session):
        headers = _register_and_login(client)
        response = client.get(
            f"{API_PREFIX}/conversations/00000000-0000-0000-0000-000000000000",
            headers=headers,
        )
        assert response.status_code == 404

    def test_get_conversation_detail_exposes_retrieval_metadata(self, client, db_session):
        headers = _register_and_login(client)
        user_id = _user_id(db_session, "jane@example.com")
        conversation = _seed_conversation(db_session, user_id, title="Meta")
        MessageRepository(db_session).create(
            Message(
                conversation_id=conversation.id,
                user_id=user_id,
                role=MessageRole.ASSISTANT,
                content="Answer.",
                retrieval_metadata=[
                    {
                        "document_id": str(uuid.uuid4()),
                        "filename": "doc.txt",
                        "page_number": 3,
                        "chunk_index": 1,
                        "score": 0.91,
                        "snippet": "snippet text",
                    }
                ],
            )
        )

        response = client.get(f"{API_PREFIX}/conversations/{conversation.id}", headers=headers)

        assert response.status_code == 200
        message = response.json()["messages"][0]
        assert message["role"] == "assistant"
        assert message["content"] == "Answer."
        assert message["retrieval_metadata"][0]["filename"] == "doc.txt"
        assert message["retrieval_metadata"][0]["page_number"] == 3
        assert message["retrieval_metadata"][0]["chunk_index"] == 1
        assert message["retrieval_metadata"][0]["score"] == 0.91
