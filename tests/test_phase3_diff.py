"""Phase 3 — Diff preview tests for write_file and edit_file."""

from pathlib import Path

from devbot.tools import write_file, edit_file, _compute_diff


# ============================================================================
# 1. _compute_diff unit tests
# ============================================================================

class TestComputeDiff:
    """Unit tests for the _compute_diff helper."""

    def test_no_changes_returns_no_changes_note(self):
        result = _compute_diff("hello\n", "hello\n", "test.txt")
        assert result == "(no changes)"

    def test_simple_change_returns_fenced_diff(self):
        result = _compute_diff(
            "line1\nline2\nline3\n",
            "line1\nCHANGED\nline3\n",
            "test.txt",
        )
        assert result.startswith("```diff")
        assert result.endswith("```")
        assert "-line2" in result
        assert "+CHANGED" in result
        assert "a/test.txt" in result or "---" in result

    def test_splitlines_keepends_matters(self):
        """Verify that trailing newlines are preserved in the diff."""
        result = _compute_diff(
            "a\nb\nc\n",    # has trailing newline
            "a\nb\nc",      # no trailing newline
            "f",
        )
        # Should show the removal of the trailing newline
        assert result != "(no changes)"
        assert "```diff" in result

    def test_diff_clips_large_output(self):
        """A very large diff is clipped and a truncation note appears."""
        old = "\n".join(f"line{i}" for i in range(500))
        new = "\n".join(f"LINE{i}" for i in range(500))
        result = _compute_diff(old, new, "big.txt")
        assert "... [diff truncated]" in result
        assert result.startswith("```diff")


# ============================================================================
# 2. edit_file diff tests
# ============================================================================

class TestEditFileDiff:
    """edit_file returns a unified diff after a successful edit."""

    def test_edit_file_returns_diff(self, tmp_path: Path):
        f = tmp_path / "greeting.txt"
        f.write_text("hello world\n")
        result = edit_file("greeting.txt", "hello", "goodbye", tmp_path)
        # Original prefix still intact
        assert result.startswith("Replaced 1 occurrence(s)")
        # Diff block is appended
        assert "```diff" in result
        assert "-hello" in result
        assert "+goodbye" in result

    def test_edit_file_no_diff_when_no_change(self, tmp_path: Path):
        """Contrived: replacing with same string returns (no changes)."""
        f = tmp_path / "same.txt"
        f.write_text("unchanged\n")
        result = edit_file("same.txt", "unchanged", "unchanged", tmp_path)
        assert "Replaced 1 occurrence(s)" in result
        assert "(no changes)" in result

    def test_edit_file_prefix_preserved(self, tmp_path: Path):
        """The original 'Replaced ... occurrence(s) in ...' prefix is always present."""
        f = tmp_path / "prefix.txt"
        f.write_text("alpha beta gamma\n")
        result = edit_file("prefix.txt", "beta", "delta", tmp_path)
        assert result.startswith("Replaced 1 occurrence(s) in")
        assert str(tmp_path / "prefix.txt") in result

    def test_edit_file_error_no_diff_appended(self, tmp_path: Path):
        """When old_string is not found, the error message has no diff block."""
        f = tmp_path / "missing.txt"
        f.write_text("content\n")
        result = edit_file("missing.txt", "nope", "replacement", tmp_path)
        assert result.startswith("Error:")
        assert "```diff" not in result

    def test_edit_file_replace_all_diff(self, tmp_path: Path):
        """replace_all still returns a diff."""
        f = tmp_path / "multi.txt"
        f.write_text("x x x\n")
        result = edit_file("multi.txt", "x", "y", tmp_path, replace_all=True)
        assert result.startswith("Replaced 3 occurrence(s)")
        assert "```diff" in result


# ============================================================================
# 3. write_file diff tests
# ============================================================================

class TestWriteFileDiff:
    """write_file returns a diff for existing files, line-count note for new files."""

    def test_write_existing_file_returns_diff(self, tmp_path: Path):
        f = tmp_path / "existing.txt"
        f.write_text("old content\n")
        result = write_file(str(f), "new content\n", tmp_path)
        # Original prefix preserved
        assert "Wrote" in result
        assert "chars" in result
        assert "```diff" in result
        assert "-old content" in result
        assert "+new content" in result

    def test_write_new_file_returns_line_count_not_diff(self, tmp_path: Path):
        f = tmp_path / "brand_new.txt"
        # File does not exist yet
        result = write_file(str(f), "line one\nline two\nline three\n", tmp_path)
        assert "Wrote" in result
        assert "chars" in result
        # No diff block for new files
        assert "```diff" not in result
        # Should mention line count and "new file"
        assert "3 lines" in result
        assert "new file" in result

    def test_write_new_file_single_line(self, tmp_path: Path):
        f = tmp_path / "single.txt"
        result = write_file(str(f), "just one line", tmp_path)
        assert "1 lines" in result
        assert "new file" in result

    def test_write_file_prefix_preserved(self, tmp_path: Path):
        """The original 'Wrote ... chars to ...' prefix is always present."""
        f = tmp_path / "prefix.txt"
        f.write_text("before\n")
        result = write_file(str(f), "after\n", tmp_path)
        assert result.startswith("Wrote ")
        assert "chars to" in result
        assert str(tmp_path / "prefix.txt") in result

    def test_write_existing_no_content_change(self, tmp_path: Path):
        """Rewriting identical content shows (no changes)."""
        f = tmp_path / "unchanged.txt"
        f.write_text("same\n")
        result = write_file(str(f), "same\n", tmp_path)
        assert "(no changes)" in result
        assert "Wrote" in result

    def test_write_file_diff_clipped(self, tmp_path: Path):
        """A large write diff is clipped."""
        f = tmp_path / "huge.txt"
        old_lines = "\n".join(f"old{i}" for i in range(500))
        f.write_text(old_lines)
        new_lines = "\n".join(f"NEW{i}" for i in range(500))
        result = write_file(str(f), new_lines, tmp_path)
        assert "Wrote" in result
        assert "... [diff truncated]" in result
