"""
search.py - Search and Query Processing for the Search Engine

Implements the 'print' and 'find' commands:
  - print <word>: Display the full inverted index entry for a word
  - find <query>: Return ranked list of pages containing all query terms

For multi-word queries, only pages containing ALL terms are returned
(AND logic), ranked by combined TF-IDF score.
"""

from typing import Any


def print_word(index: dict, word: str) -> None:
    """
    Print the full inverted index entry for a given word.
    Shows every page the word appears in, with frequency, positions,
    and TF-IDF score.

    Args:
        index: The loaded inverted index.
        word: The word to look up (case-insensitive).
    """
    word = word.lower().strip()

    if not word:
        print("[SEARCH] Error: no word provided. Usage: print <word>")
        return

    if word not in index:
        print(f"[SEARCH] '{word}' not found in index.")
        return

    pages = index[word]
    print(f"\n=== Index entry for '{word}' ===")
    print(f"Found in {len(pages)} page(s):\n")

    # Sort pages by TF-IDF score descending for readability
    sorted_pages = sorted(
        pages.items(),
        key=lambda x: x[1].get("tf_idf", 0),
        reverse=True
    )

    for url, stats in sorted_pages:
        print(f"  URL      : {url}")
        print(f"  Frequency: {stats['frequency']}")
        print(f"  Positions: {stats['positions']}")
        print(f"  TF-IDF   : {stats.get('tf_idf', 'N/A')}")
        print()


def find_pages(index: dict, query: str) -> list[tuple[str, float]]:
    """
    Find all pages containing every word in the query (AND logic).
    Results are ranked by combined TF-IDF score (highest first).

    For a single word query, returns pages containing that word.
    For multi-word queries, returns only pages containing ALL words,
    ranked by the sum of TF-IDF scores across all query terms.

    Args:
        index: The loaded inverted index.
        query: One or more space-separated search terms.

    Returns:
        A list of (url, combined_score) tuples, sorted by score descending.
        Returns an empty list if no pages match or query is empty.
    """
    # Tokenise the query the same way we tokenised the pages
    words = [w.lower().strip() for w in query.split() if w.strip()]

    if not words:
        print("[SEARCH] Error: empty query.")
        return []

    # Check which words exist in the index
    missing = [w for w in words if w not in index]
    if missing:
        print(f"[SEARCH] The following word(s) were not found in index: {missing}")
        return []

    # Start with the set of pages containing the first word
    # Then intersect with pages for each subsequent word (AND logic)
    matching_pages = set(index[words[0]].keys())
    for word in words[1:]:
        matching_pages &= set(index[word].keys())

    if not matching_pages:
        print(f"[SEARCH] No pages found containing all of: {words}")
        return []

    # Rank results by combined TF-IDF score across all query words
    scored_results: list[tuple[str, float]] = []
    for url in matching_pages:
        combined_score = sum(
            index[word][url].get("tf_idf", 0)
            for word in words
        )
        scored_results.append((url, round(combined_score, 6)))

    # Sort by score descending (most relevant first)
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return scored_results


def print_find_results(results: list[tuple[str, float]], query: str) -> None:
    """
    Pretty-print the results of a find query to the terminal.

    Args:
        results: List of (url, score) tuples from find_pages().
        query: The original query string (for display purposes).
    """
    if not results:
        return

    print(f"\n=== Results for '{query}' ===")
    print(f"Found {len(results)} matching page(s):\n")

    for rank, (url, score) in enumerate(results, start=1):
        print(f"  {rank}. {url}")
        print(f"     Relevance score: {score}")
        print()


def search(index: dict, query: str) -> None:
    """
    Combined entry point: find pages and print results.

    Args:
        index: The loaded inverted index.
        query: One or more space-separated search terms.
    """
    results = find_pages(index, query)
    print_find_results(results, query)