"""
chunker.py — Splits documents into chunks for embedding.

Strategy: RecursiveCharacterTextSplitter with metadata preservation.
Chunk size 512 tokens (~2048 chars) with 10% overlap.

Why these numbers:
- text-embedding-3-small has 8191 token limit, but smaller chunks
  improve retrieval precision for Q&A tasks
- 10% overlap prevents context loss at chunk boundaries
- We'll experiment with these params in Sprint 2 notebooks
"""

import re
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.loader import Document
import logging

logger = logging.getLogger(__name__)


CHUNK_SIZE = 1000        # characters (not tokens)
CHUNK_OVERLAP = 100      # characters
MIN_CHUNK_LENGTH = 50    # discard noise chunks shorter than this


@dataclass
class Chunk:
    """A single embeddable unit of text with full provenance metadata."""
    chunk_id: str          # deterministic: {source_slug}-{index}
    content: str
    source_url: str
    title: str
    section: str           # beginner | intermediate | advanced
    chunk_index: int
    total_chunks: int


def _clean_text(text: str) -> str:
    """
    Normalize whitespace and remove encoding artifacts.
    Keeps code blocks intact — collapses only excessive blank lines.
    """
    # Remove Sphinx pilcrow encoding artifacts
    text = re.sub(r'Â¶|¶', '', text)
    # Collapse 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces 
    text = re.sub(r'[ \t]{3,}', '  ', text)
    return text.strip()


def _make_slug(url: str) -> str:
    """Convert URL to a stable, filesystem-safe identifier."""
    slug = url.split("python.org/3/")[-1]
    slug = re.sub(r'[^a-z0-9]', '-', slug.lower())
    return slug.strip('-')


def chunk_document(doc: Document) -> list[Chunk]:
    """
    Split a single Document into overlapping Chunks.

    Args:
        doc: Parsed Document from loader.py

    Returns:
        List of Chunk objects with metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Split hierarchy: paragraphs → sentences → words → chars
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    clean_content = _clean_text(doc.content)
    raw_chunks = splitter.split_text(clean_content)

    # Filter noise
    raw_chunks = [c for c in raw_chunks if len(c.strip()) >= MIN_CHUNK_LENGTH]

    slug = _make_slug(doc.source_url)
    total = len(raw_chunks)

    chunks = []
    for i, text in enumerate(raw_chunks):
        chunks.append(Chunk(
            chunk_id=f"{slug}-{i:04d}",
            content=text.strip(),
            source_url=doc.source_url,
            title=doc.title,
            section=doc.section,
            chunk_index=i,
            total_chunks=total,
        ))

    logger.info(f"'{doc.title}' → {total} chunks")
    return chunks


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    """
    Chunk all documents. Entry point for the ingestion pipeline.

    Args:
        docs: List of Documents from load_python_docs()

    Returns:
        Flat list of all Chunks across all documents
    """
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

    logger.info(f"Total chunks produced: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.ingestion.loader import load_python_docs

    docs = load_python_docs()
    chunks = chunk_documents(docs)

    print(f"\nChunking summary:")
    print(f"  Documents: {len(docs)}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Avg chunks/doc: {len(chunks)/len(docs):.1f}")
    print(f"\nSample chunk:")
    print(f"  ID: {chunks[0].chunk_id}")
    print(f"  Section: {chunks[0].section}")
    print(f"  Length: {len(chunks[0].content)} chars")
    print(f"  Preview: {chunks[0].content[:200]}...")
