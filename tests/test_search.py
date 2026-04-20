"""
test_search.py - Unit Tests for the Search Module

Tests cover:
  - print_word(): found words, missing words, empty input
  - find_pages(): single word, multi-word AND logic, ranking, edge cases
  - search(): combined entry point
"""

import pytest
from io import StringIO
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from search import print_word, find_pages, search


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_index():
    """A small sample inverted index for testing."""
    return {
        "hello": {
            "http://page1.com": {"frequency": 3, "positions": [0, 5, 10], "tf_idf": 0.5},
            "http://page2.com": {"frequency": 1, "positions": [2], "tf_idf": 0.1},
        },
        "world": {
            "http://page1.com": {"frequency": 2, "positions": [1, 6], "tf_idf": 0.4},
            "http://page3.com": {"frequency": 1, "positions": [0], "tf_idf": 0.2},
        },
        "python": {
            "http://page2.com": {"frequency": 5, "positions": [0, 1, 2, 3, 4], "tf_idf": 0.9},
        },
        "good": {
            "http://page1.com": {"frequency": 1, "positions": [3], "tf_idf": 0.3},
            "http://page2.com": {"frequency": 2, "positions": [1, 4], "tf_idf": 0.6},
        },
        "friends": {
            "http://page1.com": {"frequency": 1, "positions": [7], "tf_idf": 0.25},
            "http://page2.com": {"frequency": 1, "positions": [9], "tf_idf": 0.25},
        },
    }


# ---------------------------------------------------------------------------
# print_word() tests
# ---------------------------------------------------------------------------

class TestPrintWord:

    def test_prints_word_found_in_index(self, sample_index, capsys):
        """Should print stats when word exists in index."""
        print_word(sample_index, "hello")
        captured = capsys.readouterr()

        assert "hello" in captured.out
        assert "http://page1.com" in captured.out
        assert "http://page2.com" in captured.out

    def test_prints_frequency(self, sample_index, capsys):
        """Output should include frequency information."""
        print_word(sample_index, "hello")
        captured = capsys.readouterr()

        assert "Frequency" in captured.out or "frequency" in captured.out.lower()

    def test_word_not_found(self, sample_index, capsys):
        """Should print a not-found message for unknown words."""
        print_word(sample_index, "nonexistent")
        captured = capsys.readouterr()

        assert "not found" in captured.out.lower() or "nonexistent" in captured.out

    def test_case_insensitive_lookup(self, sample_index, capsys):
        """Should find 'Hello' and 'HELLO' the same as 'hello'."""
        print_word(sample_index, "Hello")
        captured_1 = capsys.readouterr()

        print_word(sample_index, "HELLO")
        captured_2 = capsys.readouterr()

        assert "http://page1.com" in captured_1.out
        assert "http://page1.com" in captured_2.out

    def test_empty_word_prints_usage(self, sample_index, capsys):
        """Should print usage hint for empty input."""
        print_word(sample_index, "")
        captured = capsys.readouterr()

        assert "usage" in captured.out.lower() or "error" in captured.out.lower()

    def test_whitespace_only_word(self, sample_index, capsys):
        """Should treat whitespace-only input as empty."""
        print_word(sample_index, "   ")
        captured = capsys.readouterr()

        assert "usage" in captured.out.lower() or "error" in captured.out.lower()


# ---------------------------------------------------------------------------
# find_pages() tests
# ---------------------------------------------------------------------------

class TestFindPages:

    def test_single_word_returns_matching_pages(self, sample_index):
        """Should return all pages containing a single word."""
        results = find_pages(sample_index, "hello")
        urls = [r[0] for r in results]

        assert "http://page1.com" in urls
        assert "http://page2.com" in urls

    def test_single_word_not_in_index(self, sample_index, capsys):
        """Should return empty list and print message for unknown word."""
        results = find_pages(sample_index, "nonexistent")

        assert results == []

    def test_multi_word_and_logic(self, sample_index):
        """Multi-word query should return only pages containing ALL words."""
        results = find_pages(sample_index, "good friends")
        urls = [r[0] for r in results]

        # Both page1 and page2 have 'good' and 'friends'
        assert "http://page1.com" in urls
        assert "http://page2.com" in urls
        # page3 doesn't have 'good', so should not appear
        assert "http://page3.com" not in urls

    def test_multi_word_no_intersection(self, sample_index):
        """Should return empty list when no page contains all query words."""
        # 'python' is only on page2, 'world' is on page1 and page3 — no overlap
        results = find_pages(sample_index, "python world")

        assert results == []

    def test_results_ranked_by_tf_idf(self, sample_index):
        """Results should be sorted highest TF-IDF score first."""
        results = find_pages(sample_index, "hello")

        # page1 has tf_idf=0.5, page2 has tf_idf=0.1, so page1 should be first
        assert results[0][0] == "http://page1.com"

    def test_returns_scores_with_results(self, sample_index):
        """Each result should be a (url, score) tuple."""
        results = find_pages(sample_index, "hello")

        for item in results:
            assert len(item) == 2
            url, score = item
            assert isinstance(url, str)
            assert isinstance(score, float)

    def test_empty_query_returns_empty(self, sample_index):
        """Should return empty list for empty query string."""
        results = find_pages(sample_index, "")

        assert results == []

    def test_whitespace_query_returns_empty(self, sample_index):
        """Should return empty list for whitespace-only query."""
        results = find_pages(sample_index, "   ")

        assert results == []

    def test_case_insensitive_query(self, sample_index):
        """Query should work regardless of case."""
        results_lower = find_pages(sample_index, "hello")
        results_upper = find_pages(sample_index, "HELLO")
        results_mixed = find_pages(sample_index, "Hello")

        assert [r[0] for r in results_lower] == [r[0] for r in results_upper]
        assert [r[0] for r in results_lower] == [r[0] for r in results_mixed]

    def test_one_word_missing_returns_empty(self, sample_index):
        """If any word in a multi-word query is missing, return empty list."""
        results = find_pages(sample_index, "hello nonexistent")

        assert results == []

    def test_combined_score_is_sum_of_tf_idf(self, sample_index):
        """Combined score for multi-word query should be sum of per-word TF-IDF."""
        results = find_pages(sample_index, "good friends")
        result_map = {url: score for url, score in results}

        # page1: good=0.3, friends=0.25 → 0.55
        expected_page1 = round(0.3 + 0.25, 6)
        assert abs(result_map["http://page1.com"] - expected_page1) < 0.001


# ---------------------------------------------------------------------------
# search() integration tests
# ---------------------------------------------------------------------------

class TestSearch:

    def test_search_prints_results(self, sample_index, capsys):
        """search() should print results to stdout."""
        search(sample_index, "hello")
        captured = capsys.readouterr()

        assert "http://page1.com" in captured.out

    def test_search_empty_query(self, sample_index, capsys):
        """search() with empty query should print an error, not crash."""
        search(sample_index, "")
        captured = capsys.readouterr()

        assert captured.out != ""  # Something was printed

    def test_search_missing_word(self, sample_index, capsys):
        """search() with unknown word should print a not-found message."""
        search(sample_index, "zzznonsense")
        captured = capsys.readouterr()

        assert captured.out != ""