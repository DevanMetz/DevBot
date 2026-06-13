"""Tests for devbot.gitcheckpoint — git checkpoint helpers."""

import subprocess

import pytest

from devbot.gitcheckpoint import current_branch, is_clean, ensure_branch, commit_all


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_repo(path, branch="main"):
    """Create a git repo at *path* with one commit so HEAD is defined.

    Uses ``git init --initial-branch=<branch>``, configures a dummy
    user, and commits an empty ``.gitkeep`` so the repo is never in a
    detached/empty-HEAD state.
    """
    subprocess.run(
        ["git", "init", "--initial-branch", branch, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    # Configure a fake identity so commits work.
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(path),
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=str(path),
        capture_output=True,
        text=True,
        check=True,
    )
    # In a truly fresh repo without any commits, ``git rev-parse HEAD``
    # fails and ``git status --porcelain`` shows everything as untracked.
    # Give the repo one commit so the distinction between "clean" and
    # "dirty" is predictable.
    (path / ".gitkeep").write_text("")
    subprocess.run(
        ["git", "add", "-A"],
        cwd=str(path),
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=str(path),
        capture_output=True,
        text=True,
        check=True,
    )


def _branch(path) -> str:
    """Return the current branch name (raw git output)."""
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(path),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# ---------------------------------------------------------------------------
# current_branch
# ---------------------------------------------------------------------------


class TestCurrentBranch:
    """Tests for ``current_branch``."""

    def test_returns_main_on_fresh_repo(self, tmp_path):
        _init_repo(tmp_path)
        assert current_branch(str(tmp_path)) == "main"

    def test_returns_master_when_init_with_master(self, tmp_path):
        _init_repo(tmp_path, branch="master")
        assert current_branch(str(tmp_path)) == "master"

    def test_returns_new_branch_after_checkout(self, tmp_path):
        _init_repo(tmp_path)
        subprocess.run(
            ["git", "checkout", "-b", "feature-x"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            check=True,
        )
        assert current_branch(str(tmp_path)) == "feature-x"

    def test_raises_on_non_git_directory(self, tmp_path):
        non_git = tmp_path / "not_a_repo"
        non_git.mkdir()
        with pytest.raises(RuntimeError, match="git rev-parse failed"):
            current_branch(str(non_git))


# ---------------------------------------------------------------------------
# is_clean
# ---------------------------------------------------------------------------


class TestIsClean:
    """Tests for ``is_clean``."""

    def test_clean_after_commit(self, tmp_path):
        _init_repo(tmp_path)
        # Only the initial .gitkeep commit — clean.
        assert is_clean(str(tmp_path)) is True

    def test_dirty_with_untracked_file(self, tmp_path):
        _init_repo(tmp_path)
        (tmp_path / "new_file.txt").write_text("hello")
        assert is_clean(str(tmp_path)) is False

    def test_dirty_with_modified_tracked_file(self, tmp_path):
        _init_repo(tmp_path)
        (tmp_path / "tracked.txt").write_text("original")
        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "add tracked"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            check=True,
        )
        assert is_clean(str(tmp_path)) is True
        # Now modify
        (tmp_path / "tracked.txt").write_text("modified")
        assert is_clean(str(tmp_path)) is False

    def test_raises_on_non_git_directory(self, tmp_path):
        non_git = tmp_path / "not_a_repo"
        non_git.mkdir()
        with pytest.raises(RuntimeError, match="git status failed"):
            is_clean(str(non_git))

    def test_becomes_clean_after_commit(self, tmp_path):
        _init_repo(tmp_path)
        (tmp_path / "file.txt").write_text("content")
        assert is_clean(str(tmp_path)) is False
        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "commit file"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            check=True,
        )
        assert is_clean(str(tmp_path)) is True


# ---------------------------------------------------------------------------
# ensure_branch
# ---------------------------------------------------------------------------


class TestEnsureBranch:
    """Tests for ``ensure_branch``."""

    def test_creates_and_checks_out_new_branch(self, tmp_path):
        _init_repo(tmp_path)
        ensure_branch(str(tmp_path), "feature")
        assert _branch(tmp_path) == "feature"

    def test_already_existing_branch_just_checks_out(self, tmp_path):
        _init_repo(tmp_path)
        # Create feature branch & commit something, then go back to main.
        subprocess.run(
            ["git", "checkout", "-b", "feature"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            check=True,
        )
        (tmp_path / "on_feature.txt").write_text("data")
        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "feature work"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "main"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            check=True,
        )
        assert _branch(tmp_path) == "main"

        # Now call ensure_branch with the already-existing name.
        ensure_branch(str(tmp_path), "feature")
        assert _branch(tmp_path) == "feature"

    def test_refuses_main(self, tmp_path):
        _init_repo(tmp_path)
        with pytest.raises(ValueError, match="Refusing to operate on branch"):
            ensure_branch(str(tmp_path), "main")

    def test_refuses_master(self, tmp_path):
        _init_repo(tmp_path, branch="master")
        with pytest.raises(ValueError, match="Refusing to operate on branch"):
            ensure_branch(str(tmp_path), "master")


# ---------------------------------------------------------------------------
# commit_all
# ---------------------------------------------------------------------------


class TestCommitAll:
    """Tests for ``commit_all``."""

    def test_returns_short_sha_on_success(self, tmp_path):
        _init_repo(tmp_path)
        (tmp_path / "work.txt").write_text("hello")
        sha = commit_all(str(tmp_path), "first real commit")
        assert isinstance(sha, str)
        assert len(sha) >= 3  # short SHA is typically 7+ chars, but be flexible
        # Verify it's really a commit SHA.
        stdout = subprocess.run(
            ["git", "rev-parse", sha],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert len(stdout) == 40  # full SHA

    def test_returns_none_when_nothing_to_commit(self, tmp_path):
        _init_repo(tmp_path)
        # Already clean — nothing to commit.
        result = commit_all(str(tmp_path), "should be none")
        assert result is None

    def test_new_sha_after_modification(self, tmp_path):
        _init_repo(tmp_path)
        (tmp_path / "data.txt").write_text("v1")
        sha1 = commit_all(str(tmp_path), "v1")
        assert sha1 is not None

        # Modify and commit again.
        (tmp_path / "data.txt").write_text("v2")
        sha2 = commit_all(str(tmp_path), "v2")
        assert sha2 is not None
        assert sha2 != sha1

    def test_special_characters_in_message(self, tmp_path):
        """Commit messages with quotes, newlines, and other special chars work."""
        _init_repo(tmp_path)
        (tmp_path / "file.txt").write_text("test")
        msg = '''Fix "the thing" -- it's done

More detail on line two with 'single quotes' and $dollar signs.'''
        sha = commit_all(str(tmp_path), msg)
        assert sha is not None
        # Verify the message survived the trip.
        stdout = subprocess.run(
            ["git", "log", "-1", "--format=%B", sha],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        # Git normalises trailing newlines; compare with trailing ws stripped.
        assert stdout.strip() == msg.strip()
