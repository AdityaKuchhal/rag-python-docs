"""
loader.py — Fetches and parses Python documentation pages.

Strategy: scrape docs.python.org for a curated set of topics
covering beginner → advanced Python. Returns clean text chunks
ready for the embedding pipeline.
"""

import re
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Curated Python docs URLs — beginner to advanced
PYTHON_DOCS_URLS = [
    # Beginner
    "https://docs.python.org/3/tutorial/introduction.html",
    "https://docs.python.org/3/tutorial/controlflow.html",
    "https://docs.python.org/3/tutorial/datastructures.html",
    "https://docs.python.org/3/tutorial/modules.html",
    "https://docs.python.org/3/tutorial/inputoutput.html",
    "https://docs.python.org/3/tutorial/errors.html",
    "https://docs.python.org/3/tutorial/classes.html",
    # Intermediate
    "https://docs.python.org/3/tutorial/stdlib.html",
    "https://docs.python.org/3/tutorial/stdlib2.html",
    "https://docs.python.org/3/howto/functional.html",
    "https://docs.python.org/3/howto/descriptor.html",
    # Advanced
    "https://docs.python.org/3/reference/datamodel.html",
    "https://docs.python.org/3/library/asyncio-task.html",
    "https://docs.python.org/3/library/asyncio-eventloop.html",
    "https://docs.python.org/3/library/concurrent.futures.html",
    "https://docs.python.org/3/library/itertools.html",
    "https://docs.python.org/3/library/functools.html",
    "https://docs.python.org/3/library/typing.html",
    "https://docs.python.org/3/library/contextlib.html",
    "https://docs.python.org/3/glossary.html",
    "https://docs.python.org/3/reference/compound_stmts.html#function",
]


@dataclass
class Document:
    """Represents a single loaded document before chunking."""
    content: str
    source_url: str
    title: str
    section: str  # beginner | intermediate | advanced


def _classify_section(url: str) -> str:
    """Infer difficulty section from URL pattern."""
    if "/tutorial/" in url:
        path = url.split("/tutorial/")[1]
        beginner_pages = {"introduction", "controlflow", "datastructures",
                          "modules", "inputoutput", "errors", "classes"}
        page_name = path.replace(".html", "")
        return "beginner" if page_name in beginner_pages else "intermediate"
    return "advanced"


def _parse_page(html: str, url: str) -> Optional[Document]:
    """
    Extract clean text from a Python docs HTML page.
    Strips nav, sidebar, footer — keeps only article body.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove noise elements
    for tag in soup.find_all(["nav", "footer", "script", "style"]):
        tag.decompose()
    for tag in soup.find_all(class_=["sphinxsidebar", "related", "footer"]):
        tag.decompose()

    # Extract title
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Unknown"

    # Extract main content
    body = soup.find("div", role="main") or soup.find("div", class_="body")
    if not body:
        logger.warning(f"No main content found for {url}")
        return None

    content = body.get_text(separator="\n", strip=True)

    # Clean encoding artifacts from Sphinx pilcrow symbols
    content = re.sub(r'Â¶|¶', '', content)
    title = re.sub(r'Â¶|¶', '', title).strip()

    # Basic quality filter — skip near-empty pages
    if len(content) < 200:
        logger.warning(f"Content too short for {url}, skipping")
        return None

    return Document(
        content=content,
        source_url=url,
        title=title,
        section=_classify_section(url),
    )


def load_python_docs(
    urls: list[str] = PYTHON_DOCS_URLS,
    timeout: int = 10,
) -> list[Document]:
    """
    Fetch and parse Python documentation pages.

    Args:
        urls: List of docs.python.org URLs to scrape
        timeout: HTTP request timeout in seconds

    Returns:
        List of parsed Document objects
    """
    documents = []
    headers = {"User-Agent": "rag-python-docs/1.0 (educational project)"}

    for url in urls:
        try:
            logger.info(f"Fetching: {url}")
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            response.encoding = "utf-8"  # force UTF-8 — prevents Latin-1 mis-decode of em-dashes

            doc = _parse_page(response.text, url)
            if doc:
                documents.append(doc)
                logger.info(f"Loaded: {doc.title} ({len(doc.content)} chars)")

        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            continue

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs = load_python_docs()
    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  [{doc.section}] {doc.title} — {len(doc.content)} chars")
