"""
Document list/get query latency benchmark.

Seeds ~1000 documents for one user in an in-memory SQLite database (the same
approach tests/conftest.py uses for fast, isolated data-layer work), then
benchmarks the two hot document queries: paginated listing (GET /documents
backed by DocumentRepository.list_for_user) and single-document fetch (backed
by get_by_id_for_user). No external infrastructure required.

Run: pytest benchmarks/test_document_query_latency.py --benchmark-only
"""
import uuid

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.core.constants import FileType, ProcessingStatus
from src.db.base_class import Base
from src.models.document_model import Document
from src.repositories.document_repository import DocumentRepository

DOCUMENT_COUNT = 1000


@pytest.fixture(scope="module")
def seeded() -> tuple[DocumentRepository, uuid.UUID, uuid.UUID]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = Session()

    user_id = uuid.uuid4()
    session.add_all(
        Document(
            id=uuid.uuid4(),
            user_id=user_id,
            filename=f"document-{index}.txt",
            file_type=FileType.TXT,
            file_path=f"/uploads/document-{index}.txt",
            file_size_bytes=2048,
            chunk_count=8,
            processing_status=ProcessingStatus.COMPLETED,
        )
        for index in range(DOCUMENT_COUNT)
    )
    session.commit()

    existing_id = session.execute(
        select(Document.id).where(Document.user_id == user_id).limit(1)
    ).scalar_one()

    return DocumentRepository(session), user_id, existing_id


@pytest.mark.parametrize(
    "limit,offset",
    [
        (10, 0),      # first page
        (50, 100),    # mid-list page
        (50, 950),    # deep pagination
    ],
)
def test_list_for_user_pagination(benchmark, seeded, limit, offset):
    repo, user_id, _ = seeded

    def run():
        return repo.list_for_user(user_id, limit=limit, offset=offset)

    documents, total = benchmark(run)
    benchmark.extra_info["rows_seeded"] = DOCUMENT_COUNT
    assert total == DOCUMENT_COUNT
    assert len(documents) == min(limit, DOCUMENT_COUNT - offset)


def test_get_by_id_for_user(benchmark, seeded):
    repo, user_id, document_id = seeded

    def run():
        return repo.get_by_id_for_user(document_id, user_id)

    document = benchmark(run)
    benchmark.extra_info["rows_seeded"] = DOCUMENT_COUNT
    assert document is not None
    assert document.id == document_id
