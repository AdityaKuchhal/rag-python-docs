"""
embedder.py — Embeds chunks and upserts to Pinecone.

Model: text-embedding-3-small (1536 dims, $0.02/1M tokens)
Strategy: batch embed → batch upsert with progress tracking
Rate limiting: 100 chunks/batch to stay within OpenAI limits
"""

import os
import time
import logging
from dotenv import load_dotenv
from tqdm import tqdm
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from src.ingestion.chunker import Chunk

load_dotenv()
logger = logging.getLogger(__name__)

# Batch sizes
EMBED_BATCH_SIZE = 100     # chunks per OpenAI embedding call
UPSERT_BATCH_SIZE = 100    # vectors per Pinecone upsert call
RETRY_DELAY = 2            # seconds to wait on rate limit hit


def get_embeddings_client() -> OpenAIEmbeddings:
    """Initialize OpenAI embeddings with correct model."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )


def get_pinecone_index():
    """Initialize and return the Pinecone index."""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "python-docs-rag")

    if not api_key:
        raise ValueError("PINECONE_API_KEY not set in environment")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    # Verify connection
    stats = index.describe_index_stats()
    logger.info(f"Pinecone index '{index_name}' — {stats.total_vector_count} vectors currently stored")
    return index


def embed_chunks(
    chunks: list[Chunk],
    embeddings_client: OpenAIEmbeddings,
) -> list[tuple[Chunk, list[float]]]:
    """
    Embed all chunks in batches.

    Returns:
        List of (chunk, embedding_vector) tuples
    """
    results = []

    batches = [
        chunks[i:i + EMBED_BATCH_SIZE]
        for i in range(0, len(chunks), EMBED_BATCH_SIZE)
    ]

    logger.info(f"Embedding {len(chunks)} chunks in {len(batches)} batches")

    for batch_num, batch in enumerate(tqdm(batches, desc="Embedding")):
        texts = [chunk.content for chunk in batch]

        try:
            vectors = embeddings_client.embed_documents(texts)
            results.extend(zip(batch, vectors))

        except Exception as e:
            logger.error(f"Batch {batch_num} failed: {e}")
            logger.info(f"Retrying batch {batch_num} after {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

            # Single retry
            try:
                vectors = embeddings_client.embed_documents(texts)
                results.extend(zip(batch, vectors))
            except Exception as retry_e:
                logger.error(f"Batch {batch_num} retry failed, skipping: {retry_e}")
                continue

    logger.info(f"Successfully embedded {len(results)}/{len(chunks)} chunks")
    return results


def upsert_to_pinecone(
    embedded_chunks: list[tuple[Chunk, list[float]]],
    index,
) -> int:
    """
    Upsert embedded chunks to Pinecone with metadata.

    Metadata stored per vector:
    - content: the raw text (for retrieval display)
    - source_url: original docs page
    - title: document title
    - section: beginner | intermediate | advanced
    - chunk_index: position within parent document
    - total_chunks: total chunks from parent document

    Returns:
        Number of vectors successfully upserted
    """
    # Build Pinecone upsert payload
    vectors = []
    for chunk, embedding in embedded_chunks:
        vectors.append({
            "id": chunk.chunk_id,
            "values": embedding,
            "metadata": {
                "content": chunk.content,
                "source_url": chunk.source_url,
                "title": chunk.title,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
            }
        })

    # Upsert in batches
    total_upserted = 0
    batches = [
        vectors[i:i + UPSERT_BATCH_SIZE]
        for i in range(0, len(vectors), UPSERT_BATCH_SIZE)
    ]

    logger.info(f"Upserting {len(vectors)} vectors in {len(batches)} batches")

    for batch in tqdm(batches, desc="Upserting to Pinecone"):
        try:
            index.upsert(vectors=batch)
            total_upserted += len(batch)
        except Exception as e:
            logger.error(f"Upsert batch failed: {e}")
            continue

    logger.info(f"Upserted {total_upserted}/{len(vectors)} vectors")
    return total_upserted


def run_ingestion_pipeline(chunks: list[Chunk]) -> dict:
    """
    Full ingestion pipeline: embed → upsert.
    Entry point called by scripts/ingest.py

    Returns:
        Summary dict with counts and status
    """
    embeddings_client = get_embeddings_client()
    index = get_pinecone_index()

    # Embed
    embedded_chunks = embed_chunks(chunks, embeddings_client)

    # Upsert
    total_upserted = upsert_to_pinecone(embedded_chunks, index)

    # Final index stats
    stats = index.describe_index_stats()

    summary = {
        "chunks_processed": len(chunks),
        "chunks_embedded": len(embedded_chunks),
        "vectors_upserted": total_upserted,
        "total_vectors_in_index": stats.total_vector_count,
        "status": "success" if total_upserted > 0 else "failed",
    }

    return summary
