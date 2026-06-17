"""Git checkpoint helpers for DevBot.

Lightweight wrappers around ``git`` subprocess calls so the autopilot can
safely branch, commit, and query repository state without external
dependencies.
"""

import subprocess
from typing import Optional


def current_branch(root: str) -> str:
    """Return the current branch name of the git repo at *root*.

    Raises
    ------
    RuntimeError
        If the command fails (e.g. *root* is not inside a git repository).
    """
    proc = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git rev-parse failed in {root}: {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def is_clean(root: str) -> bool:
    """Return ``True`` if there are no uncommitted changes in *root*.

    Uses ``git status --porcelain``: an empty output means the working tree
    is clean (no untracked, modified, or staged files).
    """
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git status failed in {root}: {proc.stderr.strip()}"
        )
    return proc.stdout.strip() == ""


def ensure_branch(root: str, name: str) -> None:
    """Create and checkout branch *name*, or just checkout if it exists.

    Parameters
    ----------
    root : str
        Path to the git working tree.
    name : str
        Branch name.  Must not be ``"main"`` or ``"master"``.

    Raises
    ------
    ValueError
        If *name* is ``"main"`` or ``"master"``.
    RuntimeError
        If any git command fails unexpectedly.
    """
    if name in ("main", "master"):
        raise ValueError(
            f"Refusing to operate on branch '{name}'"
        )

    # Check whether the branch already exists (locale-robust).
    proc_ref = subprocess.run(
        ["git", "show-ref", "--verify", "-q", f"refs/heads/{name}"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    branch_exists = proc_ref.returncode == 0

    if branch_exists:
        proc = subprocess.run(
            ["git", "checkout", name],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"git checkout {name} failed in {root}: {proc.stderr.strip()}"
            )
    else:
        proc = subprocess.run(
            ["git", "checkout", "-b", name],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"git checkout -b {name} failed in {root}: {proc.stderr.strip()}"
            )


def commit_all(root: str, message: str) -> Optional[str]:
    """Stage all changes and commit with *message*.

    Parameters
    ----------
    root : str
        Path to the git working tree.
    message : str
        Commit message (may contain arbitrary characters).

    Returns
    -------
    str | None
        The short SHA of the new commit, or ``None`` if there was nothing
        to commit (clean tree / no changes).

    Raises
    ------
    RuntimeError
        If an unexpected git error occurs.
    """
    # Stage everything.
    proc_add = subprocess.run(
        ["git", "add", "-A"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc_add.returncode != 0:
        raise RuntimeError(
            f"git add failed in {root}: {proc_add.stderr.strip()}"
        )

    # Commit.
    proc_commit = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc_commit.returncode == 0:
        # Success — get the short SHA.
        proc_sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if proc_sha.returncode != 0:
            raise RuntimeError(
                f"git rev-parse failed in {root}: {proc_sha.stderr.strip()}"
            )
        return proc_sha.stdout.strip()

    # Detect "nothing to commit" — return None.
    combined = (proc_commit.stdout + proc_commit.stderr).lower()
    if "nothing to commit" in combined:
        return None

    # Some other failure.
    raise RuntimeError(
        f"git commit failed in {root}: {proc_commit.stderr.strip()}"
    )
