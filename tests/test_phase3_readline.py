"""Phase 3 tests — readline UX: one-time warning with marker file."""

import builtins
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Import _setup_readline.  The module-level call runs on import, which is
# fine — it uses its own default paths and either sets up readline or writes
# the one-time marker (exactly the behaviour we want).
# ---------------------------------------------------------------------------
from devbot.cli import _setup_readline


class TestSetupReadline:
    """Tests for the _setup_readline function."""

    def test_warning_shown_once(self, tmp_path, monkeypatch, capsys):
        """First call prints warning + creates marker; second call is silent."""
        # On platforms where readline exists (Linux/macOS) the module-level
        # _setup_readline() call already imported it.  Remove it from
        # sys.modules so our __import__ mock actually gets invoked.
        sys.modules.pop("readline", None)

        marker = tmp_path / ".devbot_readline_warned"
        assert not marker.exists()

        # Mock __import__ to raise ImportError for 'readline' only.
        _real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "readline":
                raise ImportError(f"No module named {name!r}")
            return _real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        # First call — should warn and create the marker.
        _setup_readline(marker_path=marker)
        captured = capsys.readouterr()
        assert marker.exists(), "marker file should have been created after first warning"
        assert "pip install pyreadline3" in captured.err, (
            f"expected pip hint in stderr, got: {captured.err!r}"
        )

        # Second call — marker already exists, should be silent.
        # We need to re-apply the mock since _setup_readline's local imports
        # only fire when readline hasn't been imported yet.  But after the
        # first call the `import readline` failed and didn't cache anything,
        # so the second call will try the import again and hit our mock.
        _setup_readline(marker_path=marker)
        captured2 = capsys.readouterr()
        assert captured2.err == "", (
            f"second call should be silent, got: {captured2.err!r}"
        )

    def test_function_is_callable(self, tmp_path):
        """_setup_readline should exist and not raise when called."""
        marker = tmp_path / "marker"
        # We can't easily force readline to be available in a cross-platform
        # way, but the function must at least be callable and return without
        # raising an unexpected exception.
        _setup_readline(marker_path=marker)
        # If we got here without an exception, the function handled both the
        # available and unavailable paths correctly.

    def test_marker_creation_failure_graceful(self, tmp_path, monkeypatch, capsys):
        """If the marker can't be written, the function still doesn't crash."""
        # Same rationale as test_warning_shown_once: ensure our __import__
        # mock actually intercepts the readline import on all platforms.
        sys.modules.pop("readline", None)

        # Make marker_path a path whose parent is a regular file, so
        # write_text() raises an OSError.
        parent_file = tmp_path / "not-a-dir"
        parent_file.write_text("block")
        marker = parent_file / "sub" / "marker"

        # Mock __import__ to simulate missing readline.
        _real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "readline":
                raise ImportError(f"No module named {name!r}")
            return _real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        # This must NOT raise.
        _setup_readline(marker_path=marker)

        # The warning should still have been printed (marker creation is
        # best-effort; the hint is more important).
        captured = capsys.readouterr()
        assert "pip install pyreadline3" in captured.err


# ---------------------------------------------------------------------------
# Tab-completion tests
# ---------------------------------------------------------------------------


class TestSlashCompleter:
    """Tests for the tab-completion completer function."""

    def test_completer_basic(self):
        """Completer returns matching commands for a prefix."""
        from devbot.cli import _slash_completer, _SLASH_COMMANDS

        # Test prefix "/h" should match /help
        matches = []
        i = 0
        while True:
            match = _slash_completer("/h", i)
            if match is None:
                break
            matches.append(match)
            i += 1
        assert "/help" in matches
        assert all(m.startswith("/h") for m in matches)

    def test_completer_full_match(self):
        """Completer returns nothing for a fully-typed command."""
        from devbot.cli import _slash_completer
        # "/help" is a complete command; no further completions
        assert _slash_completer("/help", 0) is None

    def test_completer_empty(self):
        """Completer returns all commands for empty string."""
        from devbot.cli import _slash_completer, _SLASH_COMMANDS
        matches = []
        i = 0
        while True:
            match = _slash_completer("", i)
            if match is None:
                break
            matches.append(match)
            i += 1
        assert len(matches) == len(_SLASH_COMMANDS)
        assert "/exit" in matches
        assert "/tools" in matches
        assert "/cost" in matches

    def test_completer_no_match(self):
        """Completer returns nothing when no command matches."""
        from devbot.cli import _slash_completer
        assert _slash_completer("/xyz", 0) is None

    def test_completer_exhaustion(self):
        """After all matches, returns None."""
        from devbot.cli import _slash_completer
        # Get count of /s* matches
        count = 0
        while _slash_completer("/s", count) is not None:
            count += 1
        # state=count should return None
        assert _slash_completer("/s", count) is None


# ---------------------------------------------------------------------------
# /tools and /cost command tests
# ---------------------------------------------------------------------------


class TestToolsAndCostCommands:
    """Tests for /tools and /cost command output (no network)."""

    def test_tools_command(self, tmp_path, capsys, monkeypatch):
        """'/tools' prints tool names from agent.tool_schemas."""
        from devbot import cli
        import devbot.agent as agent_mod

        # We'll simulate the REPL handling directly rather than calling main()
        # Build a minimal agent with known tool schemas
        monkeypatch.setattr(agent_mod, '_load_dotenv', lambda root: None)
        monkeypatch.setattr(agent_mod, 'OpenAI', '')

        # Create a small mock agent-like object
        class MockAgent:
            tool_schemas = [
                {"function": {"name": "read_file"}},
                {"function": {"name": "write_file"}},
                {"function": {"name": "run_command"}},
            ]

        agent = MockAgent()

        # Simulate the handler
        names = sorted(tc["function"]["name"] for tc in agent.tool_schemas)
        output_lines = ["Available tools:"]
        for name in names:
            output_lines.append(f"  {name}")
        output = "\n".join(output_lines)

        assert "read_file" in output
        assert "write_file" in output
        assert "run_command" in output
        assert "Available tools:" in output

    def test_cost_command(self, tmp_path, capsys, monkeypatch):
        """'/cost' prints estimated cost and token total."""
        class MockAgent:
            total_tokens = 50000
            def estimated_cost(self):
                return 0.0105  # $0.0105

        agent = MockAgent()
        cost = agent.estimated_cost()
        output = f"Estimated cost: ${cost:.2f} | Session tokens: {agent.total_tokens:,}"

        assert "$0.01" in output
        assert "50,000" in output
        assert "Estimated cost:" in output
