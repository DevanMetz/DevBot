"""Session persistence: save/restore conversations to .devbot/session-<id>.json.

Saves after each user turn (only the main agent, not sub-agents). Supports
listing saved sessions and resuming the latest or a specific session.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent

SESSIONS_DIR = ".devbot"
SESSION_PREFIX = "session-"


def _sessions_dir(root: Path) -> Path:
    return root / SESSIONS_DIR


def _ensure_dir(root: Path) -> Path:
    d = _sessions_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_session_id() -> str:
    """Generate a unique session id from the current UTC timestamp."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    # Add microseconds for uniqueness within the same second
    micro = f"{int(time.time() * 1_000_000) % 1_000_000:06d}"
    return f"{SESSION_PREFIX}{ts}-{micro}"


def _session_path(root: Path, session_id: str) -> Path:
    return _sessions_dir(root) / f"{session_id}.json"


def save_session(agent: "Agent") -> str | None:
    """Persist the agent's conversation state to a session file.

    Only saves for the main (unlabeled) agent — sub-agents are skipped.
    Returns the session id on success, or None if skipped.
    """
    # Only persist the top-level agent, not swarm sub-agents.
    if agent.label is not None:
        return None

    sid = getattr(agent, "session_id", None)
    if sid is None:
        sid = _make_session_id()
        agent.session_id = sid

    d = _ensure_dir(agent.root)
    now = datetime.now(timezone.utc).isoformat()

    data = {
        "id": sid,
        "created": getattr(agent, "session_created", now),
        "updated": now,
        "model": agent.model,
        "swarm": agent.swarm,
        "megaswarm": agent.megaswarm,
        "auto_approve": agent.auto_approve,
        "total_tokens": getattr(agent, "total_tokens", 0),
        "last_prompt_tokens": getattr(agent, "last_prompt_tokens", 0),
        "delegation_count": getattr(agent, "delegation_count", 0),
        "token_budget": getattr(agent, "token_budget", 0),
        "messages": agent.messages,
    }

    if not hasattr(agent, "session_created"):
        agent.session_created = now

    path = d / f"{sid}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic on POSIX; best-effort on Windows
    return sid


def load_session(root: Path, session_id: str) -> dict | None:
    """Load a saved session by id. Returns the session data dict or None."""
    path = _session_path(root, session_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def list_sessions(root: Path) -> list[dict]:
    """Return metadata for all saved sessions, newest first.

    Each entry has: id, created, updated, model, swarm, megaswarm,
    total_tokens, message_count.
    """
    d = _sessions_dir(root)
    if not d.is_dir():
        return []

    results = []
    for p in sorted(d.glob(f"{SESSION_PREFIX}*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            mtime = p.stat().st_mtime
        except (json.JSONDecodeError, OSError):
            continue

        entry = {
            "id": data.get("id", p.stem),
            "created": data.get("created", ""),
            "updated": data.get("updated", ""),
            "model": data.get("model", "?"),
            "swarm": data.get("swarm", False),
            "megaswarm": data.get("megaswarm", False),
            "total_tokens": data.get("total_tokens", 0),
            "message_count": len(data.get("messages", [])),
            "_mtime": mtime,
        }
        results.append(entry)

    # Sort newest-first by created timestamp, falling back to file mtime
    results.sort(key=lambda s: s.get("created") or s["_mtime"], reverse=True)
    # Strip internal _mtime key before returning
    for r in results:
        del r["_mtime"]
    return results


def get_latest_session_id(root: Path) -> str | None:
    """Return the id of the most recent session, or None."""
    sessions = list_sessions(root)
    if sessions:
        return sessions[0]["id"]
    return None


def restore_agent(root: Path, session_id: str | None = None,
                  auto_approve: bool = False) -> "Agent | None":
    """Create an Agent restored from a saved session.

    If *session_id* is None, the latest session is used.
    Returns None if no session can be found/loaded.
    """
    from .agent import Agent  # local import avoids circular dependency

    if session_id is None:
        session_id = get_latest_session_id(root)
    if session_id is None:
        return None

    data = load_session(root, session_id)
    if data is None:
        return None

    agent = Agent(
        root=root,
        model=data.get("model"),
        auto_approve=auto_approve,
        swarm=data.get("swarm", False),
        megaswarm=data.get("megaswarm", False),
    )
    agent.messages = data.get("messages", agent.messages)
    agent.total_tokens = data.get("total_tokens", 0)
    agent.last_prompt_tokens = data.get("last_prompt_tokens", 0)
    agent.delegation_count = data.get("delegation_count", 0)
    agent.token_budget = data.get("token_budget", 0)
    agent.session_id = data.get("id", session_id)
    agent.session_created = data.get("created", "")
    return agent
