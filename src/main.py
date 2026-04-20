"""
main.py - Command-Line Interface for the Search Engine

Provides an interactive shell with four commands:
  build  - Crawl the website and build the inverted index
  load   - Load a previously built index from disk
  print  - Print the index entry for a specific word
  find   - Find pages containing one or more search terms

Usage:
    python main.py

Then type commands at the '> ' prompt.
"""

import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(__file__))

from crawler import crawl, BASE_URL
from indexer import build_index, save_index, load_index
from search import print_word, search


def cmd_build(index_container: dict) -> None:
    """
    Crawl the website, build the inverted index, and save it to disk.
    This may take several minutes due to the politeness window.

    Args:
        index_container: A dict used to store the loaded index in memory.
    """
    print(f"[SHELL] Starting build... This will take a while (6s between requests).")
    print(f"[SHELL] Target: {BASE_URL}\n")

    crawl_gen = crawl(BASE_URL)
    index = build_index(crawl_gen)
    save_index(index)

    index_container["index"] = index
    print("\n[SHELL] Build complete. Index is ready to use.")


def cmd_load(index_container: dict) -> None:
    """
    Load a previously built index from disk into memory.

    Args:
        index_container: A dict used to store the loaded index in memory.
    """
    try:
        index = load_index()
        index_container["index"] = index
        print("[SHELL] Index loaded successfully.")
    except FileNotFoundError as e:
        print(f"[SHELL] Error: {e}")


def cmd_print(index_container: dict, args: str) -> None:
    """
    Print the inverted index entry for a specific word.

    Args:
        index_container: Dict containing the loaded index.
        args: The word to look up (from user input).
    """
    if "index" not in index_container:
        print("[SHELL] No index loaded. Run 'build' or 'load' first.")
        return

    word = args.strip()
    if not word:
        print("[SHELL] Usage: print <word>")
        print("[SHELL] Example: print nonsense")
        return

    # Only single words are supported for print
    if len(word.split()) > 1:
        print("[SHELL] 'print' takes a single word. Did you mean 'find'?")
        return

    print_word(index_container["index"], word)


def cmd_find(index_container: dict, args: str) -> None:
    """
    Find pages containing all words in the query, ranked by relevance.

    Args:
        index_container: Dict containing the loaded index.
        args: One or more search terms (from user input).
    """
    if "index" not in index_container:
        print("[SHELL] No index loaded. Run 'build' or 'load' first.")
        return

    query = args.strip()
    if not query:
        print("[SHELL] Usage: find <query>")
        print("[SHELL] Examples:")
        print("[SHELL]   find indifference")
        print("[SHELL]   find good friends")
        return

    search(index_container["index"], query)


def print_help() -> None:
    """Print available commands and usage instructions."""
    print("""
Available commands:
  build               Crawl the website and build the search index
  load                Load a previously built index from disk
  print <word>        Print the index entry for a word
  find <query>        Find pages containing the search term(s)
  help                Show this help message
  exit / quit         Exit the program

Examples:
  > build
  > load
  > print nonsense
  > find indifference
  > find good friends
""")


def run_shell() -> None:
    """
    Start the interactive command-line shell.
    Reads user input in a loop and dispatches to the appropriate command.
    """
    # Shared container for the index (passed by reference to command functions)
    index_container: dict = {}

    print("=" * 50)
    print("  COMP3011 Search Engine Tool")
    print("  Target: https://quotes.toscrape.com/")
    print("=" * 50)
    print("Type 'help' for available commands.\n")

    while True:
        try:
            raw_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C or Ctrl+D gracefully
            print("\n[SHELL] Exiting.")
            break

        if not raw_input:
            continue  # Ignore empty input

        # Split input into command and arguments
        parts = raw_input.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command == "build":
            cmd_build(index_container)

        elif command == "load":
            cmd_load(index_container)

        elif command == "print":
            cmd_print(index_container, args)

        elif command == "find":
            cmd_find(index_container, args)

        elif command in ("help", "?"):
            print_help()

        elif command in ("exit", "quit"):
            print("[SHELL] Goodbye!")
            break

        else:
            print(f"[SHELL] Unknown command: '{command}'. Type 'help' for options.")


if __name__ == "__main__":
    run_shell()