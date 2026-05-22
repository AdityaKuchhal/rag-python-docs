"""
qa_chain.py — Generates answers using GPT-4o over retrieved chunks.

Pattern: Retrieval-Augmented Generation
  1. Retrieve relevant chunks from Pinecone
  2. Build a context-grounded prompt
  3. Generate answer with GPT-4o
  4. Return answer + source citations

Prompt is engineered for:
- Grounded answers (no hallucination beyond context)
- Source citation
- Honest "I don't know" when context is insufficient
"""

import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.retrieval.retriever import retrieve

load_dotenv()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Python documentation assistant. Your job is to answer
questions about Python accurately and concisely, using ONLY the context provided.

Rules:
- Answer based strictly on the provided context chunks
- If the context does not contain enough information, say: "I don't have enough
  information in the loaded documentation to answer this confidently."
- Always cite which documentation page your answer comes from
- Use code examples from the context when relevant
- Keep answers focused and practical"""


def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    if not chunks:
        return "No relevant context found."

    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[Context {i}] From: {chunk['title']} ({chunk['source_url']})")
        lines.append(f"Relevance score: {chunk['score']}")
        lines.append(chunk['content'])
        lines.append("")  # blank line between chunks

    return "\n".join(lines)


def ask(
    question: str,
    top_k: int = 5,
    filter_section: str | None = None,
    model: str = "gpt-4o",
) -> dict:
    """
    Full RAG pipeline: retrieve → augment → generate.

    Args:
        question: Natural language question about Python
        top_k: Number of context chunks to retrieve
        filter_section: Optional difficulty filter
        model: OpenAI model to use for generation

    Returns:
        Dict with keys: answer, sources, chunks_used, question
    """
    # Step 1: Retrieve
    chunks = retrieve(question, top_k=top_k, filter_section=filter_section)

    if not chunks:
        return {
            "question": question,
            "answer": "No relevant documentation found for this query.",
            "sources": [],
            "chunks_used": 0,
        }

    # Step 2: Build context-grounded prompt
    context = build_context_block(chunks)

    user_message = f"""Question: {question}

Context from Python documentation:
{context}

Please answer the question based on the context above."""

    # Step 3: Generate with GPT-4o
    logger.info(f"Generating answer with {model}")
    llm = ChatOpenAI(
        model=model,
        temperature=0,          # deterministic — this is a QA system, not creative
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)

    # Step 4: Extract unique sources
    sources = list({
        chunk["source_url"]: {
            "url": chunk["source_url"],
            "title": chunk["title"],
            "section": chunk["section"],
        }
        for chunk in chunks
    }.values())

    return {
        "question": question,
        "answer": response.content,
        "sources": sources,
        "chunks_used": len(chunks),
    }
