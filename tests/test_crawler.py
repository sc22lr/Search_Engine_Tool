"""
test_crawler.py - Unit and Integration Tests for the Crawler

Tests cover:
  - Successful page fetching
  - Error handling (timeouts, HTTP errors, network failures)
  - Link extraction (relative URLs, external URLs, duplicates)
  - Crawl behaviour (BFS order, visited tracking, politeness)
"""

import pytest
from unittest.mock import patch, MagicMock, call
import requests

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from crawler import get_page, extract_links, crawl, BASE_URL, POLITENESS_WINDOW


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_session():
    """A mock requests.Session for use in tests."""
    return MagicMock(spec=requests.Session)


@pytest.fixture
def simple_html():
    """A minimal HTML page with two internal links."""
    return """
    <html>
      <body>
        <a href="/page/2/">Page 2</a>
        <a href="/author/einstein/">Einstein</a>
        <a href="https://external.com/">External</a>
        <p>Some text content here.</p>
      </body>
    </html>
    """


# ---------------------------------------------------------------------------
# get_page() tests
# ---------------------------------------------------------------------------

class TestGetPage:

    def test_returns_html_on_success(self, mock_session):
        """Should return HTML content when request succeeds."""
        mock_response = MagicMock()
        mock_response.text = "<html><body>Hello</body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response

        result = get_page("https://quotes.toscrape.com", mock_session)

        assert result == "<html><body>Hello</body></html>"
        mock_session.get.assert_called_once_with(
            "https://quotes.toscrape.com", timeout=10
        )

    def test_returns_none_on_timeout(self, mock_session):
        """Should return None and not raise when request times out."""
        mock_session.get.side_effect = requests.exceptions.Timeout()

        result = get_page("https://quotes.toscrape.com", mock_session)

        assert result is None

    def test_returns_none_on_http_error(self, mock_session):
        """Should return None for 404 and other HTTP errors."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_session.get.return_value = mock_response

        result = get_page("https://quotes.toscrape.com/nonexistent", mock_session)

        assert result is None

    def test_returns_none_on_connection_error(self, mock_session):
        """Should return None when a connection error occurs."""
        mock_session.get.side_effect = requests.exceptions.ConnectionError()

        result = get_page("https://quotes.toscrape.com", mock_session)

        assert result is None

    def test_returns_none_on_generic_request_exception(self, mock_session):
        """Should return None for any requests exception."""
        mock_session.get.side_effect = requests.exceptions.RequestException("error")

        result = get_page("https://quotes.toscrape.com", mock_session)

        assert result is None


# ---------------------------------------------------------------------------
# extract_links() tests
# ---------------------------------------------------------------------------

class TestExtractLinks:

    def test_extracts_internal_links(self, simple_html):
        """Should extract only internal links (same base URL)."""
        links = extract_links(simple_html, BASE_URL)

        assert BASE_URL + "/page/2" in links
        assert BASE_URL + "/author/einstein" in links

    def test_excludes_external_links(self, simple_html):
        """Should not include links to external domains."""
        links = extract_links(simple_html, BASE_URL)

        assert "https://external.com/" not in links
        assert "https://external.com" not in links

    def test_resolves_relative_urls(self, simple_html):
        """Should prepend base URL to relative links."""
        links = extract_links(simple_html, BASE_URL)

        for link in links:
            assert link.startswith(BASE_URL)

    def test_no_duplicate_links(self):
        """Should not return duplicate URLs."""
        html = """
        <html><body>
          <a href="/page/2/">Link 1</a>
          <a href="/page/2/">Duplicate</a>
        </body></html>
        """
        links = extract_links(html, BASE_URL)

        assert links.count(BASE_URL + "/page/2") == 1

    def test_empty_page_returns_empty_list(self):
        """Should return empty list for page with no links."""
        links = extract_links("<html><body><p>No links here.</p></body></html>", BASE_URL)

        assert links == []

    def test_strips_trailing_slashes(self, simple_html):
        """Should strip trailing slashes for URL consistency."""
        links = extract_links(simple_html, BASE_URL)

        for link in links:
            assert not link.endswith("/")

    def test_ignores_anchor_fragments(self):
        """Should handle anchor-only hrefs gracefully."""
        html = '<html><body><a href="#">Top</a></body></html>'
        links = extract_links(html, BASE_URL)

        # '#' is not an internal link, so should be excluded
        assert BASE_URL + "#" not in links


# ---------------------------------------------------------------------------
# crawl() integration tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestCrawl:

    @patch("crawler.time.sleep")
    @patch("crawler.requests.Session")
    def test_crawl_yields_url_and_html(self, mock_session_class, mock_sleep):
        """Should yield (url, html) tuples for each crawled page."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Hello</p></body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response

        results = list(crawl(BASE_URL))

        assert len(results) >= 1
        url, html = results[0]
        assert url == BASE_URL
        assert "Hello" in html

    @patch("crawler.time.sleep")
    @patch("crawler.requests.Session")
    def test_crawl_respects_politeness_window(self, mock_session_class, mock_sleep):
        """Should call time.sleep with at least POLITENESS_WINDOW seconds."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        page1_html = f'<html><body><a href="{BASE_URL}/page/2">p2</a></body></html>'
        page2_html = "<html><body><p>Page 2</p></body></html>"

        mock_response1 = MagicMock()
        mock_response1.text = page1_html
        mock_response1.raise_for_status = MagicMock()

        mock_response2 = MagicMock()
        mock_response2.text = page2_html
        mock_response2.raise_for_status = MagicMock()

        mock_session.get.side_effect = [mock_response1, mock_response2]

        list(crawl(BASE_URL))

        # Verify sleep was called with the correct politeness window
        mock_sleep.assert_called_with(POLITENESS_WINDOW)

    @patch("crawler.time.sleep")
    @patch("crawler.requests.Session")
    def test_crawl_does_not_revisit_pages(self, mock_session_class, mock_sleep):
        """Should not crawl the same URL more than once."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        # Page that links back to itself
        html = f'<html><body><a href="{BASE_URL}">Home</a></body></html>'
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response

        results = list(crawl(BASE_URL))

        # BASE_URL should only be visited once
        urls = [r[0] for r in results]
        assert urls.count(BASE_URL) == 1

    @patch("crawler.time.sleep")
    @patch("crawler.requests.Session")
    def test_crawl_skips_failed_pages(self, mock_session_class, mock_sleep):
        """Should continue crawling if one page fails to load."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        page1_html = f'<html><body><a href="{BASE_URL}/page/2">p2</a></body></html>'

        mock_response1 = MagicMock()
        mock_response1.text = page1_html
        mock_response1.raise_for_status = MagicMock()

        # Second page fails
        mock_session.get.side_effect = [
            mock_response1,
            requests.exceptions.Timeout()
        ]

        results = list(crawl(BASE_URL))

        # Should have crawled page 1 successfully, skipped page 2
        assert len(results) == 1
        assert results[0][0] == BASE_URL