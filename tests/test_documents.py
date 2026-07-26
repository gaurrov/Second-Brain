"""
Integration tests for the document management endpoints.

The embedding model and Qdrant are mocked out here (via monkeypatch) so
these tests run fast and offline — they exercise the API/service/
repository wiring and, critically, the multi-user isolation guarantees,
not the actual ML inference or vector DB behavior. Add a separate
marked "live" test suite against a real Qdrant instance + a real
Sentence-Transformers model for end-to-end pipeline verification.
"""
import io

import pytest

from src.core.config import settings
from src.services import document_service as document_service_module

API_PREFIX = settings.API_V1_PREFIX


@pytest.fixture(autouse=True)
def _mock_vector_layer(monkeypatch):
    """
    Prevents DocumentService.delete_document from trying to reach a real
    Qdrant instance. The background ingestion pipeline itself is never
    triggered in these tests because we don't wait for/invoke background
    tasks — TestClient does not execute them by default here since we
    never call `.wait()`-equivalent; FastAPI's BackgroundTasks run
    synchronously after the response in TestClient, so we also patch the
    ingestion entry point to a no-op to avoid loading a real embedding
    model during upload tests.
    """
    monkeypatch.setattr(
        document_service_module, "VectorRepository", lambda client: _NoOpVectorRepository()
    )
    monkeypatch.setattr(document_service_module, "get_qdrant_client", lambda: None)

    from src.services import ingestion_service as ingestion_service_module

    monkeypatch.setattr(ingestion_service_module, "process_document_task", lambda document_id: None)

    # Patch the reference used inside api.deps as well, since it imports
    # process_document_task directly into its own namespace.
    from src.api import deps as deps_module

    monkeypatch.setattr(deps_module, "process_document_task", lambda document_id: None)


class _NoOpVectorRepository:
    def delete_by_document(self, document_id, user_id):
        pass


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


def _upload_txt(client, headers, content=b"Hello world, this is a test document.", filename="note.txt"):
    return client.post(
        f"{API_PREFIX}/documents/upload",
        headers=headers,
        files={"file": (filename, io.BytesIO(content), "text/plain")},
    )


class TestUpload:
    def test_upload_txt_success(self, client):
        headers = _register_and_login(client)
        response = _upload_txt(client, headers)
        assert response.status_code == 202
        body = response.json()
        assert body["filename"] == "note.txt"
        assert body["file_type"] == "txt"
        assert body["processing_status"] == "pending"
        assert body["chunk_count"] == 0

    def test_upload_rejects_unsupported_extension(self, client):
        headers = _register_and_login(client)
        response = client.post(
            f"{API_PREFIX}/documents/upload",
            headers=headers,
            files={"file": ("malware.exe", io.BytesIO(b"binary"), "application/octet-stream")},
        )
        assert response.status_code == 415

    def test_upload_rejects_empty_file(self, client):
        headers = _register_and_login(client)
        response = client.post(
            f"{API_PREFIX}/documents/upload",
            headers=headers,
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        )
        assert response.status_code == 400

    def test_upload_requires_authentication(self, client):
        response = client.post(
            f"{API_PREFIX}/documents/upload",
            files={"file": ("note.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 401


class TestListAndGet:
    def test_list_returns_only_own_documents(self, client):
        headers_a = _register_and_login(client, email="a@example.com", username="user_a")
        headers_b = _register_and_login(client, email="b@example.com", username="user_b")

        _upload_txt(client, headers_a, filename="a_doc.txt")
        _upload_txt(client, headers_b, filename="b_doc.txt")
        _upload_txt(client, headers_b, filename="b_doc2.txt")

        response_a = client.get(f"{API_PREFIX}/documents", headers=headers_a)
        response_b = client.get(f"{API_PREFIX}/documents", headers=headers_b)

        assert response_a.json()["total"] == 1
        assert response_a.json()["documents"][0]["filename"] == "a_doc.txt"
        assert response_b.json()["total"] == 2

    def test_get_document_cross_user_forbidden(self, client):
        headers_a = _register_and_login(client, email="a@example.com", username="user_a")
        headers_b = _register_and_login(client, email="b@example.com", username="user_b")

        upload_response = _upload_txt(client, headers_a, filename="secret.txt")
        document_id = upload_response.json()["id"]

        # User B must not be able to fetch User A's document.
        response = client.get(f"{API_PREFIX}/documents/{document_id}", headers=headers_b)
        assert response.status_code == 404

        # User A can fetch their own.
        response_own = client.get(f"{API_PREFIX}/documents/{document_id}", headers=headers_a)
        assert response_own.status_code == 200

    def test_get_nonexistent_document_returns_404(self, client):
        headers = _register_and_login(client)
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"{API_PREFIX}/documents/{fake_id}", headers=headers)
        assert response.status_code == 404


class TestDelete:
    def test_delete_own_document_succeeds(self, client):
        headers = _register_and_login(client)
        upload_response = _upload_txt(client, headers)
        document_id = upload_response.json()["id"]

        delete_response = client.delete(f"{API_PREFIX}/documents/{document_id}", headers=headers)
        assert delete_response.status_code == 204

        get_response = client.get(f"{API_PREFIX}/documents/{document_id}", headers=headers)
        assert get_response.status_code == 404

    def test_delete_other_users_document_forbidden(self, client):
        headers_a = _register_and_login(client, email="a@example.com", username="user_a")
        headers_b = _register_and_login(client, email="b@example.com", username="user_b")

        upload_response = _upload_txt(client, headers_a, filename="secret.txt")
        document_id = upload_response.json()["id"]

        delete_response = client.delete(f"{API_PREFIX}/documents/{document_id}", headers=headers_b)
        assert delete_response.status_code == 404

        # Confirm it's still there for the rightful owner.
        get_response = client.get(f"{API_PREFIX}/documents/{document_id}", headers=headers_a)
        assert get_response.status_code == 200
