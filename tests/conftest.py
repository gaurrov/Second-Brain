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


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run live integration tests (require a real Qdrant server and model weights).",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-live"):
        skip_live = pytest.mark.skip(
            reason="requires --run-live (real Qdrant server + model weights)"
        )
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)



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

@pytest.fixture(autouse=True)
def _disable_real_qdrant(monkeypatch):
    """
    Prevents every test that spins up TestClient(app) from making a real
    network attempt to Qdrant during the app's lifespan startup hook.
    Without this, every single test pays the cost of a real (failing)
    connection attempt, since `ensure_collection(get_qdrant_client())`
    runs on every app startup regardless of what the test is actually
    exercising.
    """
    import src.vectorstore.collection_manager as collection_manager_module
    import src.vectorstore.qdrant_client as qdrant_client_module

    from functools import lru_cache

    # Keep the lru_cache wrapper so shutdown paths that inspect it
    # (cache_info/cache_clear) keep working under test.
    monkeypatch.setattr(
        qdrant_client_module, "get_qdrant_client", lru_cache(maxsize=1)(lambda: None)
    )
    monkeypatch.setattr(collection_manager_module, "ensure_collection", lambda client: None)
