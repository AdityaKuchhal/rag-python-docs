"""
main.py — FastAPI application entry point.

Lifespan pattern used for startup/shutdown resource management.
Clients are initialized once at startup and reused across requests
— avoids re-initializing Pinecone + OpenAI on every call.
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone

from src.api.schemas import AskRequest, AskResponse, HealthResponse, Source
from src.retrieval.qa_chain import ask

load_dotenv()
logger = logging.getLogger(__name__)

# App version — increment per sprint
VERSION = "0.2.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: verify Pinecone connection and cache index stats.
    Shutdown: nothing to clean up (stateless HTTP).
    """
    logger.info("Starting RAG API...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "python-docs-rag"))
    stats = index.describe_index_stats()
    app.state.vector_count = stats.total_vector_count
    logger.info(f"Connected to Pinecone — {app.state.vector_count} vectors")
    yield
    logger.info("Shutting down RAG API")


app = FastAPI(
    title="RAG Python Docs API",
    description="AI-powered Q&A over Python documentation using RAG",
    version=VERSION,
    lifespan=lifespan,
)

# CORS — open for development, lock down in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check — verifies API is up and Pinecone is reachable."""
    return HealthResponse(
        status="ok",
        version=VERSION,
        index_vectors=app.state.vector_count,
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask_question(request: AskRequest):
    """
    Submit a natural language question about Python.

    Retrieves relevant chunks from Pinecone, generates a grounded
    answer using GPT-4o, and returns the answer with source citations.
    """
    start = time.perf_counter()

    result = ask(
        question=request.question,
        top_k=request.top_k,
        filter_section=request.section,
    )

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    sources = [
        Source(
            title=s["title"],
            url=s["url"],
            section=s["section"],
        )
        for s in result["sources"]
    ]

    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        sources=sources,
        chunks_used=result["chunks_used"],
        latency_ms=latency_ms,
    )
