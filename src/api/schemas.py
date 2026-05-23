"""
schemas.py — Pydantic request/response models for the RAG API.

Strict typing ensures FastAPI auto-validates all inputs and
generates accurate OpenAPI docs at /docs.
"""

from pydantic import BaseModel, Field
from typing import Optional


class AskRequest(BaseModel):
    """Incoming question payload."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural language question about Python",
        examples=["What is a Python decorator?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve",
    )
    section: Optional[str] = Field(
        default=None,
        description="Filter by difficulty: beginner | intermediate | advanced",
        examples=["advanced"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "How does the asyncio event loop work?",
                    "top_k": 5,
                    "section": "advanced",
                }
            ]
        }
    }


class Source(BaseModel):
    """A single source document cited in the answer."""
    title: str
    url: str
    section: str


class AskResponse(BaseModel):
    """Full response from the RAG pipeline."""
    question: str
    answer: str
    sources: list[Source]
    chunks_used: int
    latency_ms: float = Field(description="Total pipeline latency in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    index_vectors: int
