"""
Shared pytest fixtures.

Uses an in-memory SQLite database for fast, isolated unit tests of the
service/repository layers. NOTE: SQLite does not support Postgres-only
features (like server-side gen_random_uuid()), so integration tests that
need real Postgres semantics should instead point DATABASE_URL at a
disposable test Postgres instance (e.g. via docker/docker-compose.yml)
rather than relying on this fixture.
"""
import os

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-ci-only")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.db.base_class import Base
from src.db.session import get_db


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client(db_session):
    from src.main import app

    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
