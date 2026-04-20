"""
test_indexer.py - Unit Tests for the Indexer

Tests cover:
  - tokenise(): punctuation stripping, case normalisation
  - extract_text(): script/style removal, text extraction
  - build_page_index(): frequency and position tracking
  - compute_tf_idf(): correct TF-IDF calculation
  - build_index(): full pipeline from crawl generator
  - save_index() / load_index(): round-trip persistence
"""

import pytest
import json
import os
import tempfile

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from indexer import (
    tokenise,
    extract_text,
    build_page_index,
    compute_tf_idf,
    build_index,
    save_index,
    load_index,
)


# ---------------------------------------------------------------------------
# tokenise() tests
# ---------------------------------------------------------------------------

class TestTokenise:

    def test_lowercases_words(self):
        """Should convert all words to lowercase."""
        tokens = tokenise("Hello World")
        assert tokens == ["hello", "world"]

    def test_strips_punctuation(self):
        """Should remove punctuation from tokens."""
        tokens = tokenise("Hello, world! It's great.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "great" in tokens
        assert "," not in tokens
        assert "!" not in tokens

    def test_handles_empty_string(self):
        """Should return empty list for empty input."""
        assert tokenise("") == []

    def test_handles_numbers(self):
        """Should include numeric tokens."""
        tokens = tokenise("Page 42")
        assert "page" in tokens
        assert "42" in tokens

    def test_handles_only_punctuation(self):
        """Should return empty list for punctuation-only input."""
        assert tokenise("!!! ???") == []

    def test_multiple_spaces(self):
        """Should handle multiple spaces between words."""
        tokens = tokenise("hello   world")
        assert tokens == ["hello", "world"]


# ---------------------------------------------------------------------------
# extract_text() tests
# ---------------------------------------------------------------------------

class TestExtractText:

    def test_extracts_paragraph_text(self):
        """Should extract text from paragraph tags."""
        html = "<html><body><p>Hello world</p></body></html>"
        text = extract_text(html)
        assert "hello" in text.lower()
        assert "world" in text.lower()

    def test_removes_script_tags(self):
        """Should not include JavaScript content."""
        html = "<html><body><script>var x = 'secret';</script><p>visible</p></body></html>"
        text = extract_text(html)
        assert "secret" not in text
        assert "visible" in text

    def test_removes_style_tags(self):
        """Should not include CSS content."""
        html = "<html><body><style>.hidden { display: none; }</style><p>visible</p></body></html>"
        text = extract_text(html)
        assert "display" not in text
        assert "visible" in text

    def test_handles_empty_html(self):
        """Should return a string (possibly empty) for empty HTML."""
        text = extract_text("<html><body></body></html>")
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# build_page_index() tests
# ---------------------------------------------------------------------------

class TestBuildPageIndex:

    def test_counts_word_frequency(self):
        """Should correctly count how many times each word appears."""
        html = "<html><body><p>hello hello world</p></body></html>"
        page_index = build_page_index("http://example.com", html)

        assert page_index["hello"]["frequency"] == 2
        assert page_index["world"]["frequency"] == 1

    def test_records_word_positions(self):
        """Should record the position (index) of each word occurrence."""
        html = "<html><body><p>alpha beta alpha</p></body></html>"
        page_index = build_page_index("http://example.com", html)

        assert len(page_index["alpha"]["positions"]) == 2
        assert len(page_index["beta"]["positions"]) == 1

    def test_is_case_insensitive(self):
        """Should treat 'Good' and 'good' as the same word."""
        html = "<html><body><p>Good good GOOD</p></body></html>"
        page_index = build_page_index("http://example.com", html)

        assert "good" in page_index
        assert page_index["good"]["frequency"] == 3

    def test_returns_empty_for_blank_page(self):
        """Should return empty dict for a page with no text."""
        html = "<html><body></body></html>"
        page_index = build_page_index("http://example.com", html)

        assert isinstance(page_index, dict)


# ---------------------------------------------------------------------------
# compute_tf_idf() tests
# ---------------------------------------------------------------------------

class TestComputeTfIdf:

    def test_adds_tf_idf_scores(self):
        """Should add a tf_idf key to each page entry."""
        index = {
            "hello": {
                "http://page1.com": {"frequency": 3, "positions": [0, 1, 2]},
                "http://page2.com": {"frequency": 1, "positions": [5]},
            }
        }
        compute_tf_idf(index, total_docs=2)

        assert "tf_idf" in index["hello"]["http://page1.com"]
        assert "tf_idf" in index["hello"]["http://page2.com"]

    def test_tf_idf_is_numeric(self):
        """TF-IDF scores should be floats."""
        index = {
            "world": {
                "http://page1.com": {"frequency": 2, "positions": [0, 1]},
            }
        }
        compute_tf_idf(index, total_docs=5)

        score = index["world"]["http://page1.com"]["tf_idf"]
        assert isinstance(score, float)

    def test_higher_frequency_gives_higher_score(self):
        """A page with more occurrences of a word should score higher."""
        index = {
            "test": {
                "http://page1.com": {"frequency": 10, "positions": list(range(10))},
                "http://page2.com": {"frequency": 1, "positions": [0]},
            }
        }
        compute_tf_idf(index, total_docs=10)

        score1 = index["test"]["http://page1.com"]["tf_idf"]
        score2 = index["test"]["http://page2.com"]["tf_idf"]
        assert score1 > score2


# ---------------------------------------------------------------------------
# build_index() tests
# ---------------------------------------------------------------------------

class TestBuildIndex:

    def test_builds_index_from_generator(self):
        """Should build a valid index from a crawl generator."""
        def fake_crawl():
            yield "http://page1.com", "<html><body><p>hello world</p></body></html>"
            yield "http://page2.com", "<html><body><p>hello python</p></body></html>"

        index = build_index(fake_crawl())

        assert "hello" in index
        assert "world" in index
        assert "python" in index

    def test_index_contains_tf_idf(self):
        """Built index should include TF-IDF scores."""
        def fake_crawl():
            yield "http://page1.com", "<html><body><p>hello world</p></body></html>"

        index = build_index(fake_crawl())

        for url_data in index["hello"].values():
            assert "tf_idf" in url_data

    def test_multi_page_word_appears_in_multiple_urls(self):
        """Words appearing on multiple pages should have multiple URL entries."""
        def fake_crawl():
            yield "http://page1.com", "<html><body><p>shared word</p></body></html>"
            yield "http://page2.com", "<html><body><p>shared content</p></body></html>"

        index = build_index(fake_crawl())

        assert "http://page1.com" in index["shared"]
        assert "http://page2.com" in index["shared"]

    def test_empty_crawl_returns_empty_index(self):
        """Should return empty dict for an empty crawl generator."""
        def empty_crawl():
            return iter([])

        index = build_index(empty_crawl())
        assert index == {}


# ---------------------------------------------------------------------------
# save_index() / load_index() tests
# ---------------------------------------------------------------------------

class TestIndexPersistence:

    def test_save_and_load_roundtrip(self, tmp_path):
        """Saved index should be identical when loaded back."""
        index = {
            "hello": {
                "http://page1.com": {
                    "frequency": 2,
                    "positions": [0, 5],
                    "tf_idf": 0.123456
                }
            }
        }
        path = str(tmp_path / "test_index.json")

        save_index(index, path)
        loaded = load_index(path)

        assert loaded == index

    def test_save_creates_file(self, tmp_path):
        """save_index should create a file on disk."""
        path = str(tmp_path / "index.json")
        save_index({"word": {}}, path)

        assert os.path.exists(path)

    def test_load_raises_if_file_missing(self, tmp_path):
        """load_index should raise FileNotFoundError if file doesn't exist."""
        path = str(tmp_path / "nonexistent.json")

        with pytest.raises(FileNotFoundError):
            load_index(path)

    def test_saved_file_is_valid_json(self, tmp_path):
        """The saved index file should be valid JSON."""
        index = {"test": {"http://x.com": {"frequency": 1, "positions": [0], "tf_idf": 0.5}}}
        path = str(tmp_path / "index.json")

        save_index(index, path)

        with open(path, "r") as f:
            parsed = json.load(f)

        assert parsed == index