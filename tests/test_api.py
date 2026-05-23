"""
test_api.py — Integration tests for the FastAPI endpoints.

Uses TestClient (synchronous HTTPX wrapper) — no running server needed.
Tests hit the real Pinecone + OpenAI since this is an integration test suite.
Mark slow tests with @pytest.mark.slow to skip in CI if needed.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "index_vectors" in data

    def test_health_status_is_ok(self, client):
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_health_index_has_vectors(self, client):
        response = client.get("/health")
        assert response.json()["index_vectors"] > 0


class TestAskEndpoint:
    def test_ask_returns_200(self, client):
        response = client.post(
            "/ask",
            json={"question": "What is a Python list?"}
        )
        assert response.status_code == 200

    def test_ask_response_schema(self, client):
        response = client.post(
            "/ask",
            json={"question": "What is a Python list?"}
        )
        data = response.json()
        assert "question" in data
        assert "answer" in data
        assert "sources" in data
        assert "chunks_used" in data
        assert "latency_ms" in data

    def test_ask_returns_non_empty_answer(self, client):
        response = client.post(
            "/ask",
            json={"question": "How do Python decorators work?"}
        )
        data = response.json()
        assert len(data["answer"]) > 50

    def test_ask_returns_sources(self, client):
        response = client.post(
            "/ask",
            json={"question": "What is a generator in Python?"}
        )
        data = response.json()
        assert len(data["sources"]) > 0
        # Each source must have required fields
        for source in data["sources"]:
            assert "title" in source
            assert "url" in source
            assert "section" in source

    def test_ask_question_echoed_in_response(self, client):
        question = "What is a Python tuple?"
        response = client.post("/ask", json={"question": question})
        assert response.json()["question"] == question

    def test_ask_latency_is_recorded(self, client):
        response = client.post(
            "/ask",
            json={"question": "What is a Python class?"}
        )
        assert response.json()["latency_ms"] > 0

    def test_ask_with_section_filter(self, client):
        response = client.post(
            "/ask",
            json={
                "question": "How does asyncio work?",
                "section": "advanced",
                "top_k": 3,
            }
        )
        data = response.json()
        assert response.status_code == 200
        assert data["chunks_used"] <= 3
        # All sources should be advanced section
        for source in data["sources"]:
            assert source["section"] == "advanced"

    def test_ask_custom_top_k(self, client):
        response = client.post(
            "/ask",
            json={"question": "What is a Python function?", "top_k": 3}
        )
        assert response.json()["chunks_used"] <= 3


class TestValidation:
    def test_ask_rejects_empty_question(self, client):
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 422

    def test_ask_rejects_short_question(self, client):
        response = client.post("/ask", json={"question": "hi"})
        assert response.status_code == 422

    def test_ask_rejects_missing_question(self, client):
        response = client.post("/ask", json={})
        assert response.status_code == 422

    def test_ask_rejects_top_k_zero(self, client):
        response = client.post(
            "/ask",
            json={"question": "What is Python?", "top_k": 0}
        )
        assert response.status_code == 422

    def test_ask_rejects_top_k_over_limit(self, client):
        response = client.post(
            "/ask",
            json={"question": "What is Python?", "top_k": 99}
        )
        assert response.status_code == 422
