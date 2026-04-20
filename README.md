# Search Engine Tool — COMP3011 Coursework 2

A command-line search engine that crawls [quotes.toscrape.com](https://quotes.toscrape.com/), builds an inverted index with TF-IDF scoring, and allows users to search for pages by keyword.

---

## Architecture Overview

```
src/
├── crawler.py   # BFS web crawler with politeness window and error handling
├── indexer.py   # Inverted index builder with TF-IDF scoring and JSON persistence
├── search.py    # Query processing with AND logic and ranked results
└── main.py      # Interactive CLI shell
tests/
├── test_crawler.py   # Unit + integration tests for the crawler (98% coverage)
├── test_indexer.py   # Unit tests for tokenisation, indexing, TF-IDF, persistence
├── test_search.py    # Unit tests for print and find commands
└── test_main.py      # Unit tests for CLI shell commands
data/
└── index.json        # Generated inverted index (created by 'build' command)
```

### Design Rationale

- **BFS crawling** — breadth-first search ensures all pages at the same depth are crawled before going deeper, mimicking real-world crawler behaviour
- **Inverted index** — maps words → pages → statistics (frequency, positions, TF-IDF), enabling O(1) word lookups regardless of index size
- **TF-IDF ranking** — Term Frequency × Inverse Document Frequency rewards words that are frequent in a page but rare across all pages, producing more relevant search results than simple frequency counting
- **JSON storage** — human-readable format that supports easy inspection and debugging of the index
- **Generator-based pipeline** — the crawler yields pages one at a time to the indexer, keeping memory usage constant regardless of site size

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Setup

1. Clone the repository:

```bash
git clone https://github.com/sc22lr/Search_Engine_Tool.git
cd Search_Engine_Tool
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Start the interactive shell:

```bash
python src/main.py
```

### Commands

#### `build`
Crawls the entire website and builds the inverted index. Saves the index to `data/index.json`.
> ⚠️ This takes several minutes due to the required 6-second politeness window between requests.

```
> build
```

#### `load`
Loads a previously built index from `data/index.json` into memory.

```
> load
```

#### `print <word>`
Displays the full index entry for a word — all pages it appears on, with frequency, positions, and TF-IDF score.

```
> print nonsense
```

Example output:
```
=== Index entry for 'nonsense' ===
Found in 2 page(s):

  URL      : https://quotes.toscrape.com/page/3
  Frequency: 3
  Positions: [14, 27, 45]
  TF-IDF   : 0.042317
```

#### `find <query>`
Finds all pages containing every word in the query, ranked by combined TF-IDF relevance score.

```
> find indifference
> find good friends
```

Example output:
```
=== Results for 'good friends' ===
Found 2 matching page(s):

  1. https://quotes.toscrape.com/page/2
     Relevance score: 0.085
  
  2. https://quotes.toscrape.com/
     Relevance score: 0.031
```

#### `help`
Lists all available commands.

#### `exit` / `quit`
Exits the program.

---

## Testing

Install test dependencies:

```bash
pip install pytest pytest-cov
```

Run the full test suite:

```bash
python -m pytest tests/ -v
```

Run with coverage report:

```bash
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

### Coverage Summary

| Module | Coverage |
|---|---|
| crawler.py | 98% |
| indexer.py | 100% |
| search.py | 100% |
| main.py | 94% |
| **Total** | **98%** |

### Testing Strategy

- **Unit tests** — each function is tested in isolation using `unittest.mock` to patch external dependencies (HTTP requests, file I/O)
- **Integration tests** — the full crawl → index → search pipeline is tested using fake crawl generators
- **Edge cases** — empty queries, missing words, whitespace input, failed HTTP requests, missing index files, duplicate links
- **Mocked HTTP** — all crawler tests use mocked sessions so they run without hitting the internet, making them fast and deterministic

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| requests | 2.32.5 | HTTP requests for web crawling |
| beautifulsoup4 | 4.13.4 | HTML parsing and text extraction |
| pytest | ≥9.0 | Test runner |
| pytest-cov | ≥7.0 | Test coverage reporting |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Key Implementation Details

### Politeness Window
A minimum of 6 seconds is enforced between successive HTTP requests using `time.sleep()`, as required by the assignment brief and good web crawling practice.

### TF-IDF Scoring
Results are ranked using TF-IDF (Term Frequency–Inverse Document Frequency):

```
TF  = occurrences of word in page / total occurrences across all pages
IDF = log(total pages / (1 + pages containing word))
TF-IDF = TF × IDF
```

This penalises very common words (like "the") and rewards words that are distinctive to specific pages.

### Multi-word Queries
The `find` command uses **AND logic** — only pages containing *all* query terms are returned. Results are ranked by the sum of TF-IDF scores across all query terms.

---

## Module Lead
Dr Ammar Alsalka — University of Leeds, School of Computer Science