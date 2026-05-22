"""
retriever.py — Queries Pinecone for semantically similar chunks.

Strategy: dense vector similarity search using the same embedding
model used at ingestion (text-embedding-3-small). Returns top-k
chunks with scores and metadata.

Sprint 2 extension point: hybrid search (BM25 + dense) added here.
"""

import os
import logging
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.3  # discard low-confidence matches


def get_retrieval_clients():
    """Initialize Pinecone index and OpenAI embeddings."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "python-docs-rag"))

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    return index, embeddings


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    filter_section: str | None = None,
) -> list[dict]:
    """
    Retrieve top-k relevant chunks for a query.

    Args:
        query: Natural language question
        top_k: Number of chunks to retrieve
        score_threshold: Minimum cosine similarity score (0-1)
        filter_section: Optional filter — 'beginner'|'intermediate'|'advanced'

    Returns:
        List of dicts with keys: content, source_url, title,
        section, score, chunk_id
    """
    index, embeddings = get_retrieval_clients()

    # Embed the query using the same model as ingestion
    logger.info(f"Embedding query: '{query}'")
    query_vector = embeddings.embed_query(query)

    # Build optional metadata filter
    pinecone_filter = None
    if filter_section:
        pinecone_filter = {"section": {"$eq": filter_section}}

    # Query Pinecone
    logger.info(f"Querying Pinecone (top_k={top_k})")
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=pinecone_filter,
    )

    # Parse and filter results
    results = []
    for match in response.matches:
        if match.score < score_threshold:
            logger.debug(f"Skipping low-score match: {match.score:.3f}")
            continue

        results.append({
            "chunk_id": match.id,
            "content": match.metadata.get("content", ""),
            "source_url": match.metadata.get("source_url", ""),
            "title": match.metadata.get("title", ""),
            "section": match.metadata.get("section", ""),
            "score": round(match.score, 4),
        })

    logger.info(f"Retrieved {len(results)} chunks above threshold")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_queries = [
        "What is a Python decorator?",
        "How does asyncio work?",
        "What is the difference between a list and a tuple?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        results = retrieve(query, top_k=3)
        for i, r in enumerate(results):
            print(f"\n[{i+1}] Score: {r['score']} | {r['title']} | {r['section']}")
            print(f"    URL: {r['source_url']}")
            print(f"    Preview: {r['content'][:150]}...")
