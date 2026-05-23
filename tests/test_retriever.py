"""
test_retriever.py — Unit tests for the retrieval layer.

Tests retriever in isolation — verifies Pinecone query returns
correctly structured results without testing the LLM layer.
"""

import pytest
from src.retrieval.retriever import retrieve


class TestRetriever:
    def test_retrieve_returns_results(self):
        results = retrieve("What is a Python decorator?")
        assert len(results) > 0

    def test_retrieve_result_schema(self):
        results = retrieve("What is a Python list?", top_k=3)
        for r in results:
            assert "chunk_id" in r
            assert "content" in r
            assert "source_url" in r
            assert "title" in r
            assert "section" in r
            assert "score" in r

    def test_retrieve_scores_above_threshold(self):
        results = retrieve("How does asyncio work?", score_threshold=0.3)
        for r in results:
            assert r["score"] >= 0.3

    def test_retrieve_top_k_respected(self):
        results = retrieve("What is a Python class?", top_k=3)
        assert len(results) <= 3

    def test_retrieve_section_filter(self):
        results = retrieve(
            "How does the event loop work?",
            top_k=5,
            filter_section="advanced"
        )
        for r in results:
            assert r["section"] == "advanced"

    def test_retrieve_content_is_non_empty(self):
        results = retrieve("What is a generator?")
        for r in results:
            assert len(r["content"]) > 0

    def test_retrieve_source_urls_are_valid(self):
        results = retrieve("What is a Python module?")
        for r in results:
            assert r["source_url"].startswith("https://docs.python.org")
