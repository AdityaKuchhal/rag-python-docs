"""
ingest.py — Full ingestion pipeline runner.

Usage:
    python scripts/ingest.py

Runs: load → chunk → embed → upsert to Pinecone
"""

import logging
import json
import sys
import os

# Ensure src is importable from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.loader import load_python_docs
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import run_ingestion_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== RAG Ingestion Pipeline Starting ===")

    # Step 1: Load
    logger.info("Step 1/3: Loading Python documentation...")
    docs = load_python_docs()
    logger.info(f"Loaded {len(docs)} documents")

    # Step 2: Chunk
    logger.info("Step 2/3: Chunking documents...")
    chunks = chunk_documents(docs)
    logger.info(f"Produced {len(chunks)} chunks")

    # Step 3: Embed + Upsert
    logger.info("Step 3/3: Embedding and upserting to Pinecone...")
    summary = run_ingestion_pipeline(chunks)

    # Report
    logger.info("=== Ingestion Complete ===")
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
