"""
Unit tests for DocumentRepository.

Exercises the data layer only — no upload endpoints, file I/O, or
background processing. Validates multi-user isolation at the repository
level per project_rules.md.
"""
import uuid

import pytest

from src.core.constants import FileType, ProcessingStatus
from src.models.document_model import Document
from src.models.user_model import User
from src.repositories.document_repository import DocumentRepository


def _create_user(db_session, *, username: str, email: str) -> User:
    user = User(
        id=uuid.uuid4(),
        username=username,
        email=email,
        hashed_password="hashed",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def _create_document(db_session, *, user_id: uuid.UUID, filename: str) -> Document:
    document = Document(
        id=uuid.uuid4(),
        user_id=user_id,
        filename=filename,
        file_type=FileType.TXT,
        file_path=f"/tmp/{filename}",
        file_size_bytes=12,
    )
    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)
    return document


@pytest.fixture()
def document_repository(db_session):
    return DocumentRepository(db_session)


class TestDocumentRepositoryIsolation:
    def test_get_by_id_is_disabled(self, document_repository):
        with pytest.raises(NotImplementedError, match="get_by_id_for_user"):
            document_repository.get_by_id(uuid.uuid4())

    def test_get_by_id_for_user_returns_owned_document(
        self, db_session, document_repository
    ):
        user = _create_user(db_session, username="alice", email="alice@example.com")
        document = _create_document(db_session, user_id=user.id, filename="notes.txt")

        found = document_repository.get_by_id_for_user(document.id, user.id)
        assert found is not None
        assert found.id == document.id
        assert found.filename == "notes.txt"

    def test_get_by_id_for_user_returns_none_for_other_user(
        self, db_session, document_repository
    ):
        owner = _create_user(db_session, username="owner", email="owner@example.com")
        other = _create_user(db_session, username="other", email="other@example.com")
        document = _create_document(db_session, user_id=owner.id, filename="secret.txt")

        assert document_repository.get_by_id_for_user(document.id, other.id) is None

    def test_list_for_user_is_scoped(self, db_session, document_repository):
        user_a = _create_user(db_session, username="user_a", email="a@example.com")
        user_b = _create_user(db_session, username="user_b", email="b@example.com")

        _create_document(db_session, user_id=user_a.id, filename="a1.txt")
        _create_document(db_session, user_id=user_b.id, filename="b1.txt")
        _create_document(db_session, user_id=user_b.id, filename="b2.txt")

        docs_a, total_a = document_repository.list_for_user(user_a.id)
        docs_b, total_b = document_repository.list_for_user(user_b.id)

        assert total_a == 1
        assert len(docs_a) == 1
        assert docs_a[0].filename == "a1.txt"

        assert total_b == 2
        assert {doc.filename for doc in docs_b} == {"b1.txt", "b2.txt"}

    def test_delete_for_user_removes_only_owned_document(
        self, db_session, document_repository
    ):
        owner = _create_user(db_session, username="owner", email="owner@example.com")
        other = _create_user(db_session, username="other", email="other@example.com")
        document = _create_document(db_session, user_id=owner.id, filename="owned.txt")

        assert document_repository.delete_for_user(document.id, other.id) is False
        assert document_repository.get_by_id_for_user(document.id, owner.id) is not None

        assert document_repository.delete_for_user(document.id, owner.id) is True
        assert document_repository.get_by_id_for_user(document.id, owner.id) is None


class TestDocumentRepositoryStatus:
    def test_update_status(self, db_session, document_repository):
        user = _create_user(db_session, username="alice", email="alice@example.com")
        document = _create_document(db_session, user_id=user.id, filename="doc.txt")

        updated = document_repository.update_status(
            document,
            ProcessingStatus.COMPLETED,
            chunk_count=5,
            error_message=None,
        )

        assert updated.processing_status == ProcessingStatus.COMPLETED
        assert updated.chunk_count == 5
        assert updated.error_message is None

        failed = document_repository.update_status(
            updated,
            ProcessingStatus.FAILED,
            error_message="parse error",
        )
        assert failed.processing_status == ProcessingStatus.FAILED
        assert failed.error_message == "parse error"
