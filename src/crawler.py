"""
crawler.py - Web Crawler for the Search Engine

Crawls all pages of https://quotes.toscrape.com/, respecting a politeness
window of at least 6 seconds between requests. Returns raw page data
(URL + HTML content) for the indexer to process.
"""

import time
import requests
from bs4 import BeautifulSoup
from typing import Generator


# Base URL of the target website
BASE_URL = "https://quotes.toscrape.com"

# Minimum delay between HTTP requests (seconds) - required by the brief
POLITENESS_WINDOW = 6


def get_page(url: str, session: requests.Session) -> str | None:
    """
    Fetch a single page and return its HTML content as a string.
    Returns None if the request fails for any reason.

    Args:
        url: The full URL to fetch.
        session: A persistent requests.Session for connection reuse.

    Returns:
        HTML content as a string, or None on failure.
    """
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()  # Raises an error for 4xx/5xx responses
        return response.text
    except requests.exceptions.Timeout:
        print(f"[CRAWLER] Timeout fetching {url}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"[CRAWLER] HTTP error {e.response.status_code} for {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[CRAWLER] Failed to fetch {url}: {e}")
        return None


def extract_links(html: str, base_url: str) -> list[str]:
    """
    Parse an HTML page and extract all internal links (same domain).

    Args:
        html: Raw HTML content of the page.
        base_url: The base URL used to resolve relative links.

    Returns:
        A list of absolute URLs found on the page.
    """
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]

        # Handle relative URLs (e.g. "/page/2/") by prepending the base URL
        if href.startswith("/"):
            href = base_url + href

        # Only keep internal links (same domain)
        if href.startswith(base_url):
            # Strip trailing slashes for consistency
            href = href.rstrip("/")
            if href not in links:
                links.append(href)

    return links


def crawl(start_url: str = BASE_URL) -> Generator[tuple[str, str], None, None]:
    """
    Crawl the entire website starting from start_url using BFS.
    Yields (url, html) tuples for every page successfully fetched.

    Respects the POLITENESS_WINDOW delay between requests.

    Args:
        start_url: The URL to begin crawling from.

    Yields:
        Tuples of (url, html_content) for each crawled page.
    """
    visited: set[str] = set()
    queue: list[str] = [start_url.rstrip("/")]

    # Use a Session for connection reuse (more efficient than individual requests)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; COMP3011-Crawler/1.0)"
    })

    print(f"[CRAWLER] Starting crawl from {start_url}")

    while queue:
        url = queue.pop(0)  # BFS: take from front of queue

        # Skip if already visited
        if url in visited:
            continue

        visited.add(url)
        print(f"[CRAWLER] Fetching ({len(visited)}/{len(visited) + len(queue)}): {url}")

        html = get_page(url, session)

        if html is None:
            # Page failed to load - skip and continue crawling
            continue

        # Yield this page's data to the caller (indexer)
        yield url, html

        # Discover new links from this page and add unvisited ones to the queue
        new_links = extract_links(html, BASE_URL)
        for link in new_links:
            if link not in visited and link not in queue:
                queue.append(link)

        # Politeness window: wait before the next request
        if queue:
            print(f"[CRAWLER] Waiting {POLITENESS_WINDOW}s (politeness window)...")
            time.sleep(POLITENESS_WINDOW)

    session.close()
    print(f"[CRAWLER] Done. Crawled {len(visited)} pages.")