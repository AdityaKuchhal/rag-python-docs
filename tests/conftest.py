"""
conftest.py — Shared pytest fixtures.

The session-scoped client uses TestClient as a context manager so the
FastAPI lifespan (Pinecone startup, vector_count caching) runs before
any test that depends on app.state.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
