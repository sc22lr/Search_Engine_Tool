"""
indexer.py - Inverted Index Builder for the Search Engine

Processes raw HTML pages from the crawler and builds an inverted index.
The index stores, for each word:
  - Which pages contain it
  - How many times it appears in each page (term frequency)
  - Which positions (word offsets) it appears at in each page
  - TF-IDF score for each page (used for ranking search results)

Data structure:
{
    "word": {
        "url1": {
            "frequency": 3,
            "positions": [4, 17, 42],
            "tf_idf": 0.312
        },
        ...
    },
    ...
}
"""

import json
import math
import re
import os
from bs4 import BeautifulSoup
from typing import Any


# Path to save/load the index file
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "index.json")


def tokenise(text: str) -> list[str]:
    """
    Convert raw text into a list of lowercase tokens (words).
    Strips punctuation and converts to lowercase for case-insensitive search.

    Args:
        text: Raw text extracted from an HTML page.

    Returns:
        A list of lowercase word tokens.

    Example:
        >>> tokenise("Hello, World! It's a test.")
        ['hello', 'world', 'it', 's', 'a', 'test']
    """
    # Lowercase the text
    text = text.lower()
    # Extract only alphabetic/numeric words (strips punctuation)
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


def extract_text(html: str) -> str:
    """
    Extract all visible text from an HTML page, ignoring scripts and styles.

    Args:
        html: Raw HTML content of a page.

    Returns:
        A single string of all visible text content.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements — we don't want to index JS/CSS
    for tag in soup(["script", "style"]):
        tag.decompose()

    return soup.get_text(separator=" ")


def build_page_index(url: str, html: str) -> dict[str, dict]:
    """
    Build a partial index for a single page.
    Records frequency and positions for every word on the page.

    Args:
        url: The URL of the page being indexed.
        html: Raw HTML content of the page.

    Returns:
        A dict mapping each word to its stats for this page:
        { "word": { "frequency": N, "positions": [...] } }
    """
    text = extract_text(html)
    tokens = tokenise(text)
    page_index: dict[str, dict] = {}

    for position, word in enumerate(tokens):
        if word not in page_index:
            page_index[word] = {"frequency": 0, "positions": []}
        page_index[word]["frequency"] += 1
        page_index[word]["positions"].append(position)

    return page_index


def compute_tf_idf(index: dict[str, dict[str, dict]], total_docs: int) -> None:
    """
    Compute TF-IDF scores for every word/page combination in the index.
    Modifies the index in-place by adding a 'tf_idf' key to each entry.

    TF-IDF stands for Term Frequency - Inverse Document Frequency.
    It rewards words that appear often in a page but rarely across all pages,
    making search results more relevant.

    Formula:
        TF  = frequency of word in page / total words in page
        IDF = log(total documents / number of documents containing the word)
        TF-IDF = TF * IDF

    Args:
        index: The full inverted index (modified in-place).
        total_docs: Total number of pages crawled.
    """
    for word, pages in index.items():
        # IDF: how rare is this word across all pages?
        doc_frequency = len(pages)  # Number of pages containing this word
        idf = math.log(total_docs / (1 + doc_frequency))  # +1 avoids division by zero

        for url, stats in pages.items():
            # TF: how often does this word appear in this specific page?
            total_words_in_page = sum(
                p["frequency"] for p in index[word].values()
            )
            tf = stats["frequency"] / total_words_in_page if total_words_in_page > 0 else 0

            # Final TF-IDF score (rounded for storage efficiency)
            stats["tf_idf"] = round(tf * idf, 6)


def build_index(crawl_generator) -> dict[str, Any]:
    """
    Build the full inverted index from a crawl generator.
    Iterates over (url, html) pairs yielded by the crawler.

    Args:
        crawl_generator: A generator yielding (url, html) tuples.

    Returns:
        The complete inverted index with TF-IDF scores.
    """
    index: dict[str, dict[str, dict]] = {}
    total_docs = 0

    for url, html in crawl_generator:
        total_docs += 1
        print(f"[INDEXER] Indexing page {total_docs}: {url}")

        page_index = build_page_index(url, html)

        # Merge this page's index into the global index
        for word, stats in page_index.items():
            if word not in index:
                index[word] = {}
            index[word][url] = stats

    print(f"[INDEXER] Indexed {total_docs} pages, {len(index)} unique words.")

    # Compute TF-IDF scores now that we know the total document count
    print("[INDEXER] Computing TF-IDF scores...")
    compute_tf_idf(index, total_docs)

    return index


def save_index(index: dict, path: str = INDEX_PATH) -> None:
    """
    Serialise the inverted index to a JSON file on disk.

    Args:
        index: The inverted index to save.
        path: File path to save to (defaults to data/index.json).
    """
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    size_kb = os.path.getsize(path) / 1024
    print(f"[INDEXER] Index saved to {path} ({size_kb:.1f} KB)")


def load_index(path: str = INDEX_PATH) -> dict:
    """
    Load the inverted index from a JSON file on disk.

    Args:
        path: File path to load from (defaults to data/index.json).

    Returns:
        The loaded inverted index.

    Raises:
        FileNotFoundError: If the index file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Index file not found at '{path}'. "
            "Please run the 'build' command first."
        )

    with open(path, "r", encoding="utf-8") as f:
        index = json.load(f)

    print(f"[INDEXER] Index loaded from {path} ({len(index)} unique words)")
    return index