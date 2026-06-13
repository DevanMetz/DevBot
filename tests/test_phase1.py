"""Comprehensive unit tests for devbot.tools and devbot.agent (no network calls)."""

import os
import sys
from pathlib import Path

import pytest

from devbot.tools import (
    _resolve,
    _glob_match,
    PathError,
    read_file,
    write_file,
    edit_file,
    grep,
    find_files,
    run_command,
    dispatch,
    SKIP_DIRS,
    BLOCKED_PATTERNS,
)
from devbot.agent import _load_dotenv, Agent


# ============================================================================
# 1. _resolve
# ============================================================================

class TestResolve:
    """Tests for the _resolve sandbox function."""

    def test_relative_path_inside_root(self, tmp_path):
        """A relative path inside root resolves correctly."""
        root = tmp_path / "project"
        root.mkdir()
        (root / "foo").mkdir(parents=True)
        (root / "foo" / "bar.txt").write_text("hello")
        resolved = _resolve("foo/bar.txt", root)
        assert resolved == (root / "foo" / "bar.txt").resolve()

    def test_dotdot_escape_raises_path_error(self, tmp_path):
        """A relative path that escapes via .. raises PathError."""
        root = tmp_path / "sandbox"
        root.mkdir()
        # Create a file outside the sandbox so the path resolves cleanly
        (tmp_path / "outside.txt").write_text("secret")
        with pytest.raises(PathError, match="outside the project root"):
            _resolve("../outside.txt", root)

    def test_absolute_path_outside_root_raises_path_error(self, tmp_path):
        """An absolute path pointing outside root raises PathError."""
        root = tmp_path / "sandbox"
        root.mkdir()
        # Use an absolute path that definitely exists but is not under root
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        with pytest.raises(PathError, match="outside the project root"):
            _resolve(str(outside.resolve()), root)

    def test_absolute_path_inside_root_resolves_ok(self, tmp_path):
        """An absolute path that lies inside root resolves correctly."""
        root = tmp_path / "sandbox"
        root.mkdir()
        (root / "sub").mkdir()
        f = root / "sub" / "data.txt"
        f.write_text("data")
        resolved = _resolve(str(f.resolve()), root)
        assert resolved == f.resolve()

    def test_root_itself_is_accessible(self, tmp_path):
        """_resolve('.') against root returns root itself."""
        root = tmp_path / "sandbox"
        root.mkdir()
        resolved = _resolve(".", root)
        assert resolved == root.resolve()

    def test_symlink_escape_raises_path_error(self, tmp_path):
        """A symlink inside root that points outside raises PathError.

        Skipped gracefully on platforms / configurations where symlink creation
        is not permitted (e.g. Windows without developer mode).
        """
        if not hasattr(os, "symlink"):
            pytest.skip("os.symlink not available on this platform")

        root = tmp_path / "sandbox"
        root.mkdir()
        outside_file = tmp_path / "secret.txt"
        outside_file.write_text("secret")
        symlink_path = root / "link"

        try:
            symlink_path.symlink_to(outside_file)
        except OSError:
            pytest.skip("symlink creation not permitted (requires admin / dev mode on Windows)")

        with pytest.raises(PathError, match="outside the project root"):
            _resolve("link", root)


# ============================================================================
# 2. edit_file
# ============================================================================

class TestEditFile:
    """Tests for the edit_file function."""

    def test_empty_old_string_returns_error(self, tmp_path):
        root = tmp_path
        f = root / "test.txt"
        f.write_text("hello world")
        result = edit_file("test.txt", "", "x", root)
        assert "old_string must not be empty" in result

    def test_missing_old_string_returns_error_with_hint(self, tmp_path):
        root = tmp_path
        f = root / "test.txt"
        f.write_text("line one\nline two\nline three")
        # The first line of old_string is "two" which appears in "line two"
        result = edit_file("test.txt", "two\nmore lines", "replacement", root)
        assert "old_string not found" in result
        assert "Closest line is" in result
        assert "line two" in result

    def test_missing_old_string_no_closest_when_empty_first_line(self, tmp_path):
        """When old_string's first line is empty/whitespace-only, no hint is added."""
        root = tmp_path
        f = root / "test.txt"
        f.write_text("line one\nline two")
        # old_string starts with whitespace-only lines; lstrip().splitlines()[0] == ""
        result = edit_file("test.txt", "\n  \nreal", "x", root)
        assert "old_string not found" in result
        # first line after lstrip is empty, so no hint
        assert "Closest line is" not in result

    def test_non_unique_old_string_without_replace_all_returns_error(self, tmp_path):
        root = tmp_path
        f = root / "test.txt"
        f.write_text("duplicate\nduplicate\nunique")
        result = edit_file("test.txt", "duplicate", "replaced", root)
        assert "occurs 2 times" in result

    def test_successful_single_replacement(self, tmp_path):
        root = tmp_path
        f = root / "test.txt"
        f.write_text("before after before")
        result = edit_file("test.txt", "after", "DONE", root)
        assert "Replaced 1 occurrence" in result
        assert f.read_text() == "before DONE before"

    def test_successful_replace_all(self, tmp_path):
        root = tmp_path
        f = root / "test.txt"
        f.write_text("x x x")
        result = edit_file("test.txt", "x", "y", root, replace_all=True)
        assert "Replaced 3 occurrence" in result
        assert f.read_text() == "y y y"

    def test_non_existent_file_returns_error(self, tmp_path):
        root = tmp_path
        result = edit_file("no_such_file.txt", "old", "new", root)
        assert "is not a file" in result


# ============================================================================
# 3. read_file
# ============================================================================

class TestReadFile:
    """Tests for the read_file function."""

    def test_negative_offset_clamps_to_zero(self, tmp_path):
        root = tmp_path
        f = root / "test.txt"
        f.write_text("line1\nline2\nline3")
        result = read_file("test.txt", root, offset=-5)
        # Should show from line 1 (offset clamped to 0), all 3 lines, no header
        assert "1\tline1" in result
        assert "2\tline2" in result
        assert "3\tline3" in result
        assert "[showing lines" not in result

    def test_offset_past_eof_returns_error(self, tmp_path):
        root = tmp_path
        f = root / "test.txt"
        f.write_text("line1\nline2")
        result = read_file("test.txt", root, offset=10)
        assert "beyond file end" in result

    def test_truncation_header_appears_when_content_clipped_by_offset(self, tmp_path):
        root = tmp_path
        f = root / "test.txt"
        f.write_text("line1\nline2\nline3\nline4\nline5")
        result = read_file("test.txt", root, offset=2)
        assert "[showing lines 3-5 of 5" in result
        assert "3\tline3" in result

    def test_truncation_header_appears_when_content_exceeds_limit(self, tmp_path):
        root = tmp_path
        lines = [f"line{i}" for i in range(1, 101)]
        f = root / "test.txt"
        f.write_text("\n".join(lines))
        result = read_file("test.txt", root, offset=0, limit=50)
        assert "[showing lines 1-50 of 100" in result
        assert "51\tline51" not in result

    def test_no_truncation_header_when_entire_file_shown(self, tmp_path):
        root = tmp_path
        f = root / "test.txt"
        f.write_text("one\ntwo\nthree")
        result = read_file("test.txt", root, offset=0)
        assert "[showing lines" not in result
        assert "1\tone" in result
        assert "3\tthree" in result

    def test_limit_works_correctly(self, tmp_path):
        root = tmp_path
        lines = [f"line{i}" for i in range(1, 21)]
        f = root / "test.txt"
        f.write_text("\n".join(lines))
        result = read_file("test.txt", root, offset=5, limit=3)
        # offset 5 → lines 6,7,8 (0-based indexing)
        assert "6\tline6" in result
        assert "7\tline7" in result
        assert "8\tline8" in result
        assert "9\tline9" not in result
        assert "[showing lines 6-8 of 20" in result


# ============================================================================
# 4. grep / find_files — SKIP_DIRS and caps
# ============================================================================

class TestGrepFindFilesSkipDirs:
    """Tests that .git (and other SKIP_DIRS) are pruned from walks."""

    def test_grep_skips_git_dir(self, tmp_path):
        root = tmp_path
        (root / ".git").mkdir()
        (root / ".git" / "config").write_text("secret_token=abc123")
        (root / "src").mkdir()
        (root / "src" / "main.py").write_text("secret_token=abc123")
        result = grep(r"secret_token", root)
        # Should only find the src/main.py match, not the .git one
        assert "main.py" in result
        assert ".git" not in result

    def test_find_files_skips_git_dir(self, tmp_path):
        root = tmp_path
        (root / ".git").mkdir()
        (root / ".git" / "index.py").write_text("")
        (root / "src").mkdir()
        (root / "src" / "index.py").write_text("")
        result = find_files("*.py", root)
        assert "index.py" in result
        assert ".git" not in result


class TestGrepCap:
    """grep stops at 200 matches."""

    def test_grep_stops_at_200_matches(self, tmp_path):
        root = tmp_path
        # Create 201 files, each with one matching line
        for i in range(201):
            (root / f"file_{i:04d}.txt").write_text(f"match_me_{i}")
        result = grep(r"match_me", root)
        assert "stopped at 200 matches" in result
        # Count the actual match lines (excluding the truncation message)
        match_lines = [l for l in result.splitlines() if ":" in l and not l.startswith("...")]
        assert len(match_lines) == 200


class TestFindFilesCap:
    """find_files stops at 500 files."""

    def test_find_files_stops_at_500_files(self, tmp_path):
        root = tmp_path
        # Create 501 files matching *.txt
        for i in range(501):
            (root / f"item_{i:04d}.txt").write_text("")
        result = find_files("*.txt", root)
        assert "stopped at 500 files" in result
        file_lines = [l for l in result.splitlines() if l.endswith(".txt")]
        assert len(file_lines) == 500


# ============================================================================
# 5. run_command
# ============================================================================

class TestRunCommand:
    """Tests for the run_command sandbox."""

    def test_blocked_pattern_rejected(self, tmp_path):
        """Dangerous commands matching BLOCKED_PATTERNS are rejected."""
        result = run_command("rm -rf /", tmp_path)
        assert "command blocked" in result.lower()

    def test_safe_command_echo_succeeds(self, tmp_path):
        """A trivial safe command works on both Windows and Unix."""
        # Use a command that works on both platforms
        if sys.platform == "win32":
            result = run_command('cmd /c "echo hello"', tmp_path)
        else:
            result = run_command("echo hello", tmp_path)
        assert "hello" in result
        assert "exit code: 0" in result


# ============================================================================
# 6. dispatch
# ============================================================================

class TestDispatch:
    """Tests for the dispatch function (tool routing)."""

    def test_unknown_tool_returns_error(self, tmp_path):
        result = dispatch("non_existent_tool", {}, tmp_path)
        assert "unknown tool" in result

    def test_null_coerced_args_offset_becomes_zero(self, tmp_path):
        """JSON null for offset is coerced to 0 via a.get('offset') or 0."""
        root = tmp_path
        f = root / "data.txt"
        f.write_text("line1\nline2\nline3")
        # Simulate JSON null: json.loads("null") → None in Python
        result = dispatch("read_file", {"path": "data.txt", "offset": None}, root)
        # offset None → a.get("offset") or 0 → 0; should show from line 1, no header
        assert "1\tline1" in result
        assert "[showing lines" not in result

    def test_valid_tool_dispatch_works(self, tmp_path):
        root = tmp_path
        f = root / "note.txt"
        f.write_text("hello world")
        result = dispatch("read_file", {"path": "note.txt"}, root)
        assert "hello world" in result


# ============================================================================
# 7. agent._load_dotenv
# ============================================================================

class TestLoadDotenv:
    """Tests for the _load_dotenv helper."""

    def test_sets_env_var_when_not_already_set(self, tmp_path, monkeypatch):
        """A .env file with KEY=VALUE sets the env var when not already set."""
        # Ensure key is not set beforehand
        monkeypatch.delenv("TEST_VAR_NOT_SET", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR_NOT_SET=hello\n")
        _load_dotenv(tmp_path)
        assert os.environ["TEST_VAR_NOT_SET"] == "hello"

    def test_existing_env_var_takes_precedence(self, tmp_path, monkeypatch):
        """Existing env vars win over .env (setdefault behaviour)."""
        monkeypatch.setenv("EXISTING_VAR", "original")
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING_VAR=from_dotenv\n")
        _load_dotenv(tmp_path)
        assert os.environ["EXISTING_VAR"] == "original"

    def test_comments_and_blank_lines_ignored(self, tmp_path, monkeypatch):
        """Lines starting with # and blank lines are skipped."""
        monkeypatch.delenv("REAL_VAR", raising=False)
        monkeypatch.delenv("COMMENTED", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("# this is a comment\n\nREAL_VAR=present\n# another comment\n")
        _load_dotenv(tmp_path)
        assert os.environ.get("REAL_VAR") == "present"
        assert "COMMENTED" not in os.environ

    def test_quoted_values_stripped(self, tmp_path, monkeypatch):
        """Values wrapped in double or single quotes have quotes stripped."""
        monkeypatch.delenv("DOUBLE_QUOTED", raising=False)
        monkeypatch.delenv("SINGLE_QUOTED", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text(
            'DOUBLE_QUOTED="value with spaces"\n'
            "SINGLE_QUOTED='another value'\n"
        )
        _load_dotenv(tmp_path)
        assert os.environ["DOUBLE_QUOTED"] == "value with spaces"
        assert os.environ["SINGLE_QUOTED"] == "another value"


# ============================================================================
# 8. agent._keep_index and _compress_conversation
# ============================================================================

class TestKeepIndex:
    """Tests for Agent._keep_index and _compress_conversation keep logic."""

    @pytest.fixture(autouse=True)
    def _setup_api_key(self, monkeypatch):
        """Set a dummy API key so Agent.__init__ doesn't call SystemExit."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-dummy-key")

    def _make_agent(self, tmp_path):
        """Create an Agent with a minimal root. No network calls are made."""
        return Agent(root=tmp_path, auto_approve=True)

    def test_keep_index_finds_last_user_message(self, tmp_path):
        agent = self._make_agent(tmp_path)
        agent.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply1"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "reply2"},
        ]
        idx = agent._keep_index()
        assert idx == 3
        assert agent.messages[idx]["role"] == "user"
        assert agent.messages[idx]["content"] == "second"

    def test_keep_index_ensures_assistant_tool_pairs_stay_together(self, tmp_path):
        """The keep index (last user) ensures the entire current turn stays intact.

        A turn is: user -> assistant(tool_calls) -> tool -> tool -> assistant(final).
        Everything from the user onward must be kept so no tool messages are orphaned.
        """
        agent = self._make_agent(tmp_path)
        agent.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task1"},
            {"role": "assistant", "content": "I'll do task1", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"f"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "file content"},
            {"role": "assistant", "content": "task1 done"},
            {"role": "user", "content": "task2"},
            {"role": "assistant", "content": "doing task2", "tool_calls": [
                {"id": "tc2", "type": "function", "function": {"name": "grep", "arguments": '{"pattern":"x"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc2", "content": "no matches"},
            {"role": "assistant", "content": "task2 done"},
        ]
        idx = agent._keep_index()
        # The last user is "task2" at index 5
        assert idx == 5
        assert agent.messages[idx]["content"] == "task2"
        # Everything from idx onward is the complete turn 2 (user + assistant + tool + assistant)
        kept = agent.messages[idx:]
        assert kept[0]["role"] == "user"
        # The tool message for tc2 must be in the kept portion
        tool_ids_in_kept = [m["tool_call_id"] for m in kept if m["role"] == "tool"]
        assert "tc2" in tool_ids_in_kept
        # tc1 (from the earlier turn) is NOT in the kept portion
        assert "tc1" not in tool_ids_in_kept

    def test_compress_conversation_returns_false_when_nothing_to_compress(self, tmp_path):
        """When there is <= 1 message between system and the last user, return False."""
        agent = self._make_agent(tmp_path)
        # Only system + 1 user message → keep=1, to_compress = []
        agent.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "only message"},
        ]
        assert agent._compress_conversation() is False

        # System + user + assistant → keep=1, to_compress = []
        agent.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        assert agent._compress_conversation() is False

        # System + user + assistant + user → keep=3, to_compress = [user, assistant] (len 2 > 1)
        # But this WOULD trigger compression, which makes an API call, so we don't test
        # that path here. We only verify the "nothing to compress" early-return.
        agent.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        # keep=3, to_compress = messages[1:3] = [user(q1), assistant(a1)] → len 2
        # This would attempt compression (API call). We do NOT call it; we only
        # verify _keep_index returns 3.
        assert agent._keep_index() == 3


# ============================================================================
# 9. Recursive glob (**) support
# ============================================================================

class TestRecursiveGlob:
    """Tests for _glob_match, find_files, and grep with ** patterns."""

    # -- _glob_match unit tests -------------------------------------------------

    def test_bare_pattern_no_slash(self):
        """Bare pattern (no /) matches only the filename component."""
        assert _glob_match("foo.txt", "*.txt") is True
        assert _glob_match("subdir/foo.txt", "*.txt") is True
        assert _glob_match("subdir/foo.py", "*.txt") is False
        assert _glob_match("foo.txt", "foo.txt") is True
        assert _glob_match("foo.py", "foo.txt") is False

    def test_path_pattern_without_doublestar(self):
        """A pattern with / but no ** matches exactly that path depth."""
        assert _glob_match("devbot/foo.py", "devbot/*.py") is True
        assert _glob_match("subdir/devbot/foo.py", "devbot/*.py") is False

    def test_trailing_doublestar(self):
        """** at end matches everything beyond."""
        assert _glob_match("a/b/c/d.txt", "a/**") is True
        assert _glob_match("a/b.txt", "a/**") is True

    def test_mid_doublestar(self):
        """** in the middle matches zero or more path segments."""
        assert _glob_match("devbot/sub/foo.py", "devbot/**/*.py") is True
        assert _glob_match("devbot/foo.py", "devbot/**/*.py") is True
        assert _glob_match("other/foo.py", "devbot/**/*.py") is False

    def test_leading_doublestar(self):
        """** at the start matches any prefix."""
        assert _glob_match("a/b/c/file.py", "**/*.py") is True
        assert _glob_match("file.py", "**/*.py") is True
        assert _glob_match("file.txt", "**/*.py") is False

    def test_only_doublestar(self):
        """** alone matches everything."""
        assert _glob_match("anything.py", "**") is True
        assert _glob_match("a/b/c/d.txt", "**") is True

    def test_backslash_normalization(self):
        """Backslashes in path or pattern are normalized."""
        assert _glob_match("a\\b\\c.py", "a/b/*.py") is True
        assert _glob_match("a/b/c.py", "a\\b\\*.py") is True
        assert _glob_match("devbot\\tools.py", "devbot/*.py") is True

    # -- find_files integration tests -------------------------------------------

    def test_find_files_doublestar_finds_at_multiple_depths(self, tmp_path):
        """find_files('**/*.py', root) finds .py files at any depth."""
        (tmp_path / "top.py").write_text("")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "mid.py").write_text("")
        (tmp_path / "sub" / "deep").mkdir()
        (tmp_path / "sub" / "deep" / "bottom.py").write_text("")
        (tmp_path / "sub" / "deep" / "data.txt").write_text("")
        result = find_files("**/*.py", tmp_path)
        assert "top.py" in result
        assert "sub/mid.py" in result
        assert "sub/deep/bottom.py" in result
        assert "data.txt" not in result

    def test_find_files_exact_dir(self, tmp_path):
        """find_files('devbot/*.py', root) only matches directly in devbot/."""
        (tmp_path / "devbot").mkdir()
        (tmp_path / "devbot" / "tools.py").write_text("")
        (tmp_path / "devbot" / "sub").mkdir()
        (tmp_path / "devbot" / "sub" / "nested.py").write_text("")
        result = find_files("devbot/*.py", tmp_path)
        assert "devbot/tools.py" in result
        assert "nested.py" not in result
        assert "devbot/sub/nested.py" not in result

    def test_find_files_bare_pattern_backward_compat(self, tmp_path):
        """Bare '*.py' still works as before (backward compatibility)."""
        (tmp_path / "top.py").write_text("")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "nested.py").write_text("")
        (tmp_path / "other.txt").write_text("")
        result = find_files("*.py", tmp_path)
        assert "top.py" in result
        assert "sub/nested.py" in result
        assert "other.txt" not in result

    def test_find_files_doublestar_middle(self, tmp_path):
        """find_files('devbot/**/*.py', root) — ** in the middle."""
        (tmp_path / "devbot").mkdir()
        (tmp_path / "devbot" / "a.py").write_text("")
        (tmp_path / "devbot" / "sub").mkdir()
        (tmp_path / "devbot" / "sub" / "b.py").write_text("")
        (tmp_path / "devbot" / "sub" / "deep").mkdir()
        (tmp_path / "devbot" / "sub" / "deep" / "c.py").write_text("")
        (tmp_path / "other").mkdir()
        (tmp_path / "other" / "d.py").write_text("")
        result = find_files("devbot/**/*.py", tmp_path)
        assert "devbot/a.py" in result
        assert "devbot/sub/b.py" in result
        assert "devbot/sub/deep/c.py" in result
        assert "other/d.py" not in result

    # -- grep integration tests -------------------------------------------------

    def test_grep_with_subdir_glob(self, tmp_path):
        """grep with glob='subdir/*.txt' only matches directly in subdir/."""
        (tmp_path / "root.txt").write_text("hello")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "a.txt").write_text("hello world")
        (tmp_path / "subdir" / "nested").mkdir()
        (tmp_path / "subdir" / "nested" / "b.txt").write_text("hello")
        result = grep("hello", tmp_path, glob="subdir/*.txt")
        # Normalize Windows backslashes
        result = result.replace("\\", "/")
        assert "subdir/a.txt" in result
        assert "nested" not in result
        assert "root.txt" not in result

    def test_grep_with_doublestar_glob(self, tmp_path):
        """grep with glob='**/*.txt' matches at any depth."""
        (tmp_path / "top.txt").write_text("hello")
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "mid.txt").write_text("hello")
        (tmp_path / "a" / "b").mkdir()
        (tmp_path / "a" / "b" / "deep.txt").write_text("hello")
        (tmp_path / "a" / "b" / "other.py").write_text("hello")
        result = grep("hello", tmp_path, glob="**/*.txt")
        # Normalize Windows backslashes
        result = result.replace("\\", "/")
        assert "top.txt" in result
        assert "a/mid.txt" in result
        assert "a/b/deep.txt" in result
        assert "other.py" not in result


# ============================================================================
# 10. .gitignore-aware search helpers
# ============================================================================

class TestGitignoreHelper:
    """Tests for _parse_gitignore and _is_gitignored."""

    def test_no_gitignore_returns_empty(self, tmp_path):
        """When .gitignore doesn't exist, both lists are empty."""
        from devbot.tools import _parse_gitignore
        fp, dp = _parse_gitignore(tmp_path)
        assert fp == []
        assert dp == []

    def test_empty_gitignore_returns_empty(self, tmp_path):
        """When .gitignore is empty, both lists are empty."""
        from devbot.tools import _parse_gitignore
        (tmp_path / ".gitignore").write_text("")
        fp, dp = _parse_gitignore(tmp_path)
        assert fp == []
        assert dp == []

    def test_parses_file_and_dir_patterns(self, tmp_path):
        """File and directory patterns are correctly separated."""
        from devbot.tools import _parse_gitignore
        (tmp_path / ".gitignore").write_text(
            "*.pyc\n"
            "__pycache__/\n"
            ".env\n"
            "build/\n"
            "dist/\n"
            "# comment\n"
            "\n"
        )
        fp, dp = _parse_gitignore(tmp_path)
        assert "*.pyc" in fp
        assert ".env" in fp
        assert "__pycache__" in dp
        assert "build" in dp
        assert "dist" in dp
        # Comments and blanks are excluded
        assert "# comment" not in fp and "# comment" not in dp

    def test_is_gitignored_dir_pattern(self):
        """A file under a gitignored directory matches."""
        from devbot.tools import _is_gitignored
        assert _is_gitignored("build/output.txt", [], ["build"]) is True
        assert _is_gitignored("src/build/output.txt", [], ["build"]) is True
        assert _is_gitignored("src/main.py", [], ["build"]) is False

    def test_is_gitignored_file_pattern(self):
        """A file matching a file pattern is gitignored."""
        from devbot.tools import _is_gitignored
        assert _is_gitignored("module.pyc", ["*.pyc"], []) is True
        assert _is_gitignored("sub/module.pyc", ["*.pyc"], []) is True
        assert _is_gitignored("module.py", ["*.pyc"], []) is False

    def test_is_gitignored_both_patterns(self):
        """Combined dir and file patterns work together."""
        from devbot.tools import _is_gitignored
        assert _is_gitignored("dist/app.pyc", ["*.pyc"], ["dist"]) is True
        assert _is_gitignored("dist/app.py", ["*.pyc"], ["dist"]) is True
        assert _is_gitignored("src/app.pyc", ["*.pyc"], ["dist"]) is True
        assert _is_gitignored("src/app.py", ["*.pyc"], ["dist"]) is False


# ============================================================================
# 11. .gitignore-aware grep / find_files integration
# ============================================================================

class TestGitignoreIntegration:
    """Integration tests: grep and find_files respect .gitignore."""

    def test_grep_skips_gitignored_file(self, tmp_path):
        """grep excludes files matching .gitignore patterns."""
        (tmp_path / ".gitignore").write_text("*.pyc\n")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("TODO")
        (tmp_path / "src" / "main.pyc").write_text("TODO")  # gitignored
        result = grep(r"TODO", tmp_path)
        # Normalize backslashes
        result = result.replace("\\", "/")
        assert "main.py" in result
        assert "main.pyc" not in result

    def test_find_files_skips_gitignored_file(self, tmp_path):
        """find_files excludes files matching .gitignore patterns."""
        (tmp_path / ".gitignore").write_text("*.pyc\n")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "valid.py").write_text("")
        (tmp_path / "src" / "ignored.pyc").write_text("")  # gitignored
        result = find_files("*.py*", tmp_path)
        result = result.replace("\\", "/")
        assert "valid.py" in result
        assert "ignored.pyc" not in result

    def test_grep_respect_gitignore_false_includes_ignored(self, tmp_path):
        """With respect_gitignore=False, gitignored files are included."""
        (tmp_path / ".gitignore").write_text("*.pyc\n")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("TODO")
        (tmp_path / "src" / "main.pyc").write_text("TODO")
        result = grep(r"TODO", tmp_path, respect_gitignore=False)
        result = result.replace("\\", "/")
        assert "main.py" in result
        assert "main.pyc" in result

    def test_find_files_respect_gitignore_false_includes_ignored(self, tmp_path):
        """With respect_gitignore=False, gitignored files are included."""
        (tmp_path / ".gitignore").write_text("*.pyc\n")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "valid.py").write_text("")
        (tmp_path / "src" / "ignored.pyc").write_text("")
        result = find_files("*.py*", tmp_path, respect_gitignore=False)
        result = result.replace("\\", "/")
        assert "valid.py" in result
        assert "ignored.pyc" in result

    def test_gitignore_dir_pattern_skips_all_under(self, tmp_path):
        """A .gitignore dir pattern (build/) skips all files under that dir."""
        (tmp_path / ".gitignore").write_text("build/\n")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("hello")
        (tmp_path / "build").mkdir()
        (tmp_path / "build" / "output.txt").write_text("hello")
        result = grep(r"hello", tmp_path)
        result = result.replace("\\", "/")
        assert "src/main.py" in result
        assert "build" not in result

    def test_no_gitignore_no_effect(self, tmp_path):
        """When .gitignore doesn't exist, all files are searched normally."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("TODO")
        (tmp_path / "src" / "compiled.pyc").write_text("TODO")
        result = grep(r"TODO", tmp_path)
        result = result.replace("\\", "/")
        assert "main.py" in result
        assert "compiled.pyc" in result

    def test_dispatch_passes_respect_gitignore(self, tmp_path):
        """dispatch correctly passes respect_gitignore to grep/find_files."""
        (tmp_path / ".gitignore").write_text("*.pyc\n")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("TODO")
        (tmp_path / "src" / "main.pyc").write_text("TODO")
        # Default: respect_gitignore=True → pyc skipped
        result_default = dispatch("grep", {"pattern": "TODO", "path": "src"}, tmp_path)
        result_default = result_default.replace("\\", "/")
        assert "main.py" in result_default
        assert "main.pyc" not in result_default
        # Explicit False → pyc included
        result_include = dispatch("grep", {"pattern": "TODO", "path": "src", "respect_gitignore": False}, tmp_path)
        result_include = result_include.replace("\\", "/")
        assert "main.py" in result_include
        assert "main.pyc" in result_include
