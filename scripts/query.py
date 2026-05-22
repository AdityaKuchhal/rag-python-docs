"""
query.py — Interactive query runner for the RAG pipeline.

Usage:
    python scripts/query.py "What is a Python decorator?"
    python scripts/query.py "How does asyncio work?" --section advanced
    python scripts/query.py --interactive
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.qa_chain import ask

logging.basicConfig(level=logging.WARNING)  # suppress INFO in interactive mode


def print_result(result: dict):
    print(f"\n{'='*60}")
    print(f"Q: {result['question']}")
    print(f"{'='*60}")
    print(f"\n{result['answer']}")
    print(f"\n--- Sources ({result['chunks_used']} chunks used) ---")
    for s in result['sources']:
        print(f"  [{s['section']}] {s['title']}")
        print(f"           {s['url']}")


def interactive_mode():
    print("\nRAG Python Docs — Interactive Mode")
    print("Type your question or 'quit' to exit\n")

    while True:
        try:
            question = input("Question: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue
            result = ask(question)
            print_result(result)
        except KeyboardInterrupt:
            break

    print("\nExiting.")


def main():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--section", choices=["beginner", "intermediate", "advanced"],
                        help="Filter by difficulty section")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch interactive query mode")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of chunks to retrieve (default: 5)")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.question:
        result = ask(args.question, top_k=args.top_k, filter_section=args.section)
        print_result(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
