"""
test_main.py - Unit Tests for the CLI Shell (main.py)

Tests cover:
  - cmd_build(): builds and stores index in container
  - cmd_load(): loads index into container, handles missing file
  - cmd_print(): dispatches to print_word, handles missing index
  - cmd_find(): dispatches to search, handles missing index
  - run_shell(): command parsing and dispatch
"""

import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import cmd_build, cmd_load, cmd_print, cmd_find, print_help, run_shell


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_INDEX = {
    "hello": {
        "http://page1.com": {"frequency": 1, "positions": [0], "tf_idf": 0.5}
    }
}


# ---------------------------------------------------------------------------
# cmd_build() tests
# ---------------------------------------------------------------------------

class TestCmdBuild:

    @patch("main.save_index")
    @patch("main.build_index", return_value=SAMPLE_INDEX)
    @patch("main.crawl", return_value=iter([]))
    def test_build_stores_index_in_container(self, mock_crawl, mock_build, mock_save):
        """build should store the resulting index in the container."""
        container = {}
        cmd_build(container)

        assert "index" in container
        assert container["index"] == SAMPLE_INDEX

    @patch("main.save_index")
    @patch("main.build_index", return_value=SAMPLE_INDEX)
    @patch("main.crawl", return_value=iter([]))
    def test_build_calls_save_index(self, mock_crawl, mock_build, mock_save):
        """build should save the index to disk."""
        container = {}
        cmd_build(container)

        mock_save.assert_called_once_with(SAMPLE_INDEX)


# ---------------------------------------------------------------------------
# cmd_load() tests
# ---------------------------------------------------------------------------

class TestCmdLoad:

    @patch("main.load_index", return_value=SAMPLE_INDEX)
    def test_load_stores_index_in_container(self, mock_load):
        """load should store the index in the container."""
        container = {}
        cmd_load(container)

        assert "index" in container
        assert container["index"] == SAMPLE_INDEX

    @patch("main.load_index", side_effect=FileNotFoundError("No index file found."))
    def test_load_handles_missing_file(self, mock_load, capsys):
        """load should print an error if index file doesn't exist."""
        container = {}
        cmd_load(container)

        assert "index" not in container
        captured = capsys.readouterr()
        assert "error" in captured.out.lower() or "Error" in captured.out


# ---------------------------------------------------------------------------
# cmd_print() tests
# ---------------------------------------------------------------------------

class TestCmdPrint:

    def test_print_requires_loaded_index(self, capsys):
        """print should warn if no index is loaded."""
        cmd_print({}, "hello")
        captured = capsys.readouterr()

        assert "load" in captured.out.lower() or "build" in captured.out.lower()

    @patch("main.print_word")
    def test_print_dispatches_to_print_word(self, mock_print_word):
        """print should call print_word with the correct arguments."""
        container = {"index": SAMPLE_INDEX}
        cmd_print(container, "hello")

        mock_print_word.assert_called_once_with(SAMPLE_INDEX, "hello")

    def test_print_empty_args_shows_usage(self, capsys):
        """print with no word should show usage hint."""
        container = {"index": SAMPLE_INDEX}
        cmd_print(container, "")
        captured = capsys.readouterr()

        assert "usage" in captured.out.lower() or "Usage" in captured.out

    def test_print_multi_word_suggests_find(self, capsys):
        """print with multiple words should suggest using find instead."""
        container = {"index": SAMPLE_INDEX}
        cmd_print(container, "good friends")
        captured = capsys.readouterr()

        assert "find" in captured.out.lower()


# ---------------------------------------------------------------------------
# cmd_find() tests
# ---------------------------------------------------------------------------

class TestCmdFind:

    def test_find_requires_loaded_index(self, capsys):
        """find should warn if no index is loaded."""
        cmd_find({}, "hello")
        captured = capsys.readouterr()

        assert "load" in captured.out.lower() or "build" in captured.out.lower()

    @patch("main.search")
    def test_find_dispatches_to_search(self, mock_search):
        """find should call search with the correct arguments."""
        container = {"index": SAMPLE_INDEX}
        cmd_find(container, "hello world")

        mock_search.assert_called_once_with(SAMPLE_INDEX, "hello world")

    def test_find_empty_args_shows_usage(self, capsys):
        """find with no query should show usage hint."""
        container = {"index": SAMPLE_INDEX}
        cmd_find(container, "")
        captured = capsys.readouterr()

        assert "usage" in captured.out.lower() or "Usage" in captured.out


# ---------------------------------------------------------------------------
# print_help() tests
# ---------------------------------------------------------------------------

class TestPrintHelp:

    def test_help_mentions_all_commands(self, capsys):
        """help should mention all four main commands."""
        print_help()
        captured = capsys.readouterr()

        assert "build" in captured.out
        assert "load" in captured.out
        assert "print" in captured.out
        assert "find" in captured.out


# ---------------------------------------------------------------------------
# run_shell() integration tests
# ---------------------------------------------------------------------------

class TestRunShell:

    @patch("builtins.input", side_effect=["exit"])
    def test_exit_command_stops_loop(self, mock_input, capsys):
        """'exit' command should terminate the shell cleanly."""
        run_shell()
        captured = capsys.readouterr()

        assert "goodbye" in captured.out.lower()

    @patch("builtins.input", side_effect=["quit"])
    def test_quit_command_stops_loop(self, mock_input, capsys):
        """'quit' command should terminate the shell cleanly."""
        run_shell()
        captured = capsys.readouterr()

        assert "goodbye" in captured.out.lower()

    @patch("builtins.input", side_effect=["help", "exit"])
    def test_help_command(self, mock_input, capsys):
        """'help' command should print available commands."""
        run_shell()
        captured = capsys.readouterr()

        assert "build" in captured.out
        assert "find" in captured.out

    @patch("builtins.input", side_effect=["unknowncmd", "exit"])
    def test_unknown_command_prints_error(self, mock_input, capsys):
        """Unknown commands should print an error message."""
        run_shell()
        captured = capsys.readouterr()

        assert "unknown" in captured.out.lower()

    @patch("builtins.input", side_effect=["", "exit"])
    def test_empty_input_ignored(self, mock_input, capsys):
        """Empty input should be silently ignored."""
        run_shell()  # Should not crash

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_keyboard_interrupt_exits_gracefully(self, mock_input, capsys):
        """Ctrl+C should exit the shell without crashing."""
        run_shell()
        captured = capsys.readouterr()

        assert "exit" in captured.out.lower() or "goodbye" in captured.out.lower()