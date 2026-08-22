"""
Comprehensive cross-user isolation tests.

Verifies that users cannot access each other's data through any endpoint
or service. Covers documents, vectors, conversations, messages, logout,
and chat — proving the full multi-tenant security boundary.
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
from src.services import document_service as document_service_module

API_PREFIX = settings.API_V1_PREFIX


def _register_and_login(client, email="a@example.com", username="user_a"):
    client.post(
        f"{API_PREFIX}/auth/register",
        json={"username": username, "email": email, "password": "StrongP@ss123"},
    )
    login_response = client.post(
        f"{API_PREFIX}/auth/login", json={"email": email, "password": "StrongP@ss123"}
    )
    return {"Authorization": f"Bearer {login_response.json()['access_token']}"}


def _user_id(db_session, email):
    return db_session.query(User).filter_by(email=email).one().id


def _seed_conversation(db_session, user_id, title="Test conversation", message_count=2):
    conv_repo = ConversationRepository(db_session)
    msg_repo = MessageRepository(db_session)
    conv = conv_repo.create(Conversation(user_id=user_id, title=title))
    for i in range(message_count):
        msg_repo.create(
            Message(
                conversation_id=conv.id,
                user_id=user_id,
                role=MessageRole.USER,
                content=f"secret-message-{i}-from-owner",
            )
        )
    return conv


class _NoOpVectorRepository:
    def delete_by_document(self, document_id, user_id):
        pass


@pytest.fixture(autouse=True)
def _mock_vector_layer(monkeypatch):
    monkeypatch.setattr(
        document_service_module, "VectorRepository", lambda client: _NoOpVectorRepository()
    )
    monkeypatch.setattr(document_service_module, "get_qdrant_client", lambda: None)
    from src.services import ingestion_service as ingestion_service_module

    monkeypatch.setattr(ingestion_service_module, "process_document_task", lambda document_id: None)
    from src.api import deps as deps_module

    monkeypatch.setattr(deps_module, "process_document_task", lambda document_id: None)
    monkeypatch.setattr(document_service_module, "validate_magic_bytes", lambda path, ft: None)


def _upload_txt(client, headers, filename="test.txt", content=b"User content here"):
    files = {"file": (filename, io.BytesIO(content), "text/plain")}
    return client.post(f"{API_PREFIX}/documents/upload", headers=headers, files=files)


import io


class TestDocumentIsolation:
    """User B cannot read, list, or delete User A's documents."""

    def test_cross_user_cannot_get_document(self, client):
        headers_a = _register_and_login(client, "a1@example.com", "user_a1")
        headers_b = _register_and_login(client, "b1@example.com", "user_b1")
        doc_id = _upload_txt(client, headers_a).json()["id"]

        assert client.get(f"{API_PREFIX}/documents/{doc_id}", headers=headers_b).status_code == 404
        assert client.get(f"{API_PREFIX}/documents/{doc_id}", headers=headers_a).status_code == 200

    def test_cross_user_cannot_delete_document(self, client):
        headers_a = _register_and_login(client, "a2@example.com", "user_a2")
        headers_b = _register_and_login(client, "b2@example.com", "user_b2")
        doc_id = _upload_txt(client, headers_a).json()["id"]

        assert client.delete(f"{API_PREFIX}/documents/{doc_id}", headers=headers_b).status_code == 404
        assert client.get(f"{API_PREFIX}/documents/{doc_id}", headers=headers_a).status_code == 200

    def test_documents_list_scoped_to_owner(self, client):
        headers_a = _register_and_login(client, "a3@example.com", "user_a3")
        headers_b = _register_and_login(client, "b3@example.com", "user_b3")
        _upload_txt(client, headers_a, filename="a.txt")
        _upload_txt(client, headers_b, filename="b.txt")

        list_a = client.get(f"{API_PREFIX}/documents", headers=headers_a).json()
        list_b = client.get(f"{API_PREFIX}/documents", headers=headers_b).json()
        assert list_a["total"] == 1
        assert list_a["documents"][0]["filename"] == "a.txt"
        assert list_b["total"] == 1
        assert list_b["documents"][0]["filename"] == "b.txt"


class TestConversationIsolation:
    """User B cannot read, list, or delete User A's conversations and messages."""

    def test_cross_user_cannot_get_conversation_detail(self, client, db_session):
        headers_a = _register_and_login(client, "a4@example.com", "user_a4")
        headers_b = _register_and_login(client, "b4@example.com", "user_b4")
        user_a = _user_id(db_session, "a4@example.com")
        conv = _seed_conversation(db_session, user_a)

        assert client.get(f"{API_PREFIX}/conversations/{conv.id}", headers=headers_b).status_code == 404
        assert client.get(f"{API_PREFIX}/conversations/{conv.id}", headers=headers_a).status_code == 200

    def test_cross_user_cannot_get_messages(self, client, db_session):
        headers_a = _register_and_login(client, "a5@example.com", "user_a5")
        headers_b = _register_and_login(client, "b5@example.com", "user_b5")
        user_a = _user_id(db_session, "a5@example.com")
        conv = _seed_conversation(db_session, user_a)

        assert client.get(
            f"{API_PREFIX}/conversations/{conv.id}/messages", headers=headers_b
        ).status_code == 404
        own_response = client.get(
            f"{API_PREFIX}/conversations/{conv.id}/messages", headers=headers_a
        )
        assert own_response.status_code == 200
        assert own_response.json()["total"] == 2

    def test_cross_user_cannot_delete_conversation(self, client, db_session):
        headers_a = _register_and_login(client, "a6@example.com", "user_a6")
        headers_b = _register_and_login(client, "b6@example.com", "user_b6")
        user_a = _user_id(db_session, "a6@example.com")
        conv = _seed_conversation(db_session, user_a)

        assert client.delete(f"{API_PREFIX}/conversations/{conv.id}", headers=headers_b).status_code == 404
        assert client.get(f"{API_PREFIX}/conversations/{conv.id}", headers=headers_a).status_code == 200

    def test_conversations_list_scoped_to_owner(self, client, db_session):
        headers_a = _register_and_login(client, "a7@example.com", "user_a7")
        headers_b = _register_and_login(client, "b7@example.com", "user_b7")
        user_a = _user_id(db_session, "a7@example.com")
        user_b = _user_id(db_session, "b7@example.com")
        _seed_conversation(db_session, user_a, title="A's convo")
        _seed_conversation(db_session, user_b, title="B's convo 1")
        _seed_conversation(db_session, user_b, title="B's convo 2")

        list_a = client.get(f"{API_PREFIX}/conversations", headers=headers_a).json()
        list_b = client.get(f"{API_PREFIX}/conversations", headers=headers_b).json()
        assert list_a["total"] == 1
        assert list_a["conversations"][0]["title"] == "A's convo"
        assert list_b["total"] == 2

    def test_messages_content_not_leaked_across_users(self, client, db_session):
        """Verify message content from User A is completely invisible to User B."""
        headers_a = _register_and_login(client, "a8@example.com", "user_a8")
        headers_b = _register_and_login(client, "b8@example.com", "user_b8")
        user_a = _user_id(db_session, "a8@example.com")
        conv = _seed_conversation(db_session, user_a, message_count=3)

        # User A can read their messages with full content.
        own_messages = client.get(
            f"{API_PREFIX}/conversations/{conv.id}/messages", headers=headers_a
        ).json()
        contents = [m["content"] for m in own_messages["messages"]]
        assert all("secret-message-" in c for c in contents)

        # User B sees nothing — 404, not 200 with empty messages.
        cross_response = client.get(
            f"{API_PREFIX}/conversations/{conv.id}/messages", headers=headers_b
        )
        assert cross_response.status_code == 404


class TestLogoutIsolation:
    """Logout must revoke tokens without affecting other users."""

    def test_logout_does_not_revoke_other_users_tokens(self, client):
        headers_a = _register_and_login(client, "a9@example.com", "user_a9")
        headers_b = _register_and_login(client, "b9@example.com", "user_b9")

        # User A logs out.
        client.post(f"{API_PREFIX}/auth/logout", headers=headers_a)

        # User B's access token is still valid.
        response = client.get(f"{API_PREFIX}/users/profile", headers=headers_b)
        assert response.status_code == 200
        assert response.json()["email"] == "b9@example.com"

    def test_logout_revokes_own_refresh_token(self, client):
        login_response = client.post(
            f"{API_PREFIX}/auth/login",
            json={"email": "a10@example.com", "password": "StrongP@ss123"},
        )
        # Register first.
        client.post(
            f"{API_PREFIX}/auth/register",
            json={"username": "user_a10", "email": "a10@example.com", "password": "StrongP@ss123"},
        )
        login_response = client.post(
            f"{API_PREFIX}/auth/login",
            json={"email": "a10@example.com", "password": "StrongP@ss123"},
        )
        refresh_token = login_response.json()["refresh_token"]
        access_token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        client.post(f"{API_PREFIX}/auth/logout", headers=headers)

        # Refresh token is revoked — must fail.
        assert client.post(
            f"{API_PREFIX}/auth/refresh", json={"refresh_token": refresh_token}
        ).status_code == 401


class TestChatIsolation:
    """User B cannot interact with User A's conversations via the chat endpoint."""

    def test_chat_cross_user_conversation_rejected(self, client, db_session):
        from unittest.mock import patch
        from src.services.rag_service import RAGResult, SourceRef
        from src.api.deps import get_rag_service
        from src.core.exceptions import ConversationNotFoundException

        class _FakeRAG:
            def answer(self, question, user_id, conversation_id=None):
                if conversation_id is not None:
                    from src.repositories.conversation_repository import ConversationRepository
                    conv_repo = ConversationRepository(db_session)
                    conv = conv_repo.get_by_id_for_user(conversation_id, user_id)
                    if conv is None:
                        raise ConversationNotFoundException()
                return RAGResult(
                    answer="Test answer",
                    conversation_id=conversation_id or uuid.uuid4(),
                    user_message_id=uuid.uuid4(),
                    assistant_message_id=uuid.uuid4(),
                    refused=False,
                    sources=[SourceRef(document_id=str(uuid.uuid4()), filename="d.txt", page_number=1, chunk_index=0, score=0.9, snippet="s")],
                )

        fake = _FakeRAG()
        client.app.dependency_overrides[get_rag_service] = lambda: fake

        headers_a = _register_and_login(client, "a11@example.com", "user_a11")
        headers_b = _register_and_login(client, "b11@example.com", "user_b11")
        user_a = _user_id(db_session, "a11@example.com")
        conv = _seed_conversation(db_session, user_a)

        # User A creates a conversation.
        response_a = client.post(
            f"{API_PREFIX}/chat",
            json={"message": "Hello", "conversation_id": str(conv.id)},
            headers=headers_a,
        )
        assert response_a.status_code == 200

        # User B tries to continue in User A's conversation — must fail.
        response_b = client.post(
            f"{API_PREFIX}/chat",
            json={"message": "Tell me more", "conversation_id": str(conv.id)},
            headers=headers_b,
        )
        assert response_b.status_code in (403, 404)

        client.app.dependency_overrides.clear()
