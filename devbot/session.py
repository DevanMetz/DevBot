"""Session persistence: save/restore conversations to .devbot/session-<id>.json.

Saves after each user turn (only the main agent, not sub-agents). Supports
listing saved sessions and resuming the latest or a specific session.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent

SESSIONS_DIR = ".devbot"
SESSION_PREFIX = "session-"

# Pattern for API keys to redact in exports.  The devlog module also defines
# ^sk-[a-zA-Z0-9]+$ (anchored) for whole-value matching in structured data; we
# use a non-anchored version here so keys embedded in message text are caught.
_API_KEY_RX = re.compile(r"\bsk-[a-zA-Z0-9]+\b")


def _sessions_dir(root: Path) -> Path:
    return root / SESSIONS_DIR


def _ensure_dir(root: Path) -> Path:
    d = _sessions_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    return d


_id_counter = itertools.count()


def _make_session_id() -> str:
    """Generate a unique session id.

    Uses a UTC timestamp plus a process-wide counter and random suffix so IDs
    are unique even on platforms with coarse clock resolution (e.g. Windows,
    where time.time() granularity is ~16ms and rapid calls would otherwise
    collide).
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    uniq = f"{next(_id_counter):04d}{os.urandom(2).hex()}"
    return f"{SESSION_PREFIX}{ts}-{uniq}"


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
        "completion_tokens": getattr(agent, "completion_tokens", 0),
        "prompt_cache_hit_tokens": getattr(agent, "prompt_cache_hit_tokens", 0),
        "prompt_cache_miss_tokens": getattr(agent, "prompt_cache_miss_tokens", 0),
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
    agent.completion_tokens = data.get("completion_tokens", 0)
    agent.prompt_cache_hit_tokens = data.get("prompt_cache_hit_tokens", 0)
    agent.prompt_cache_miss_tokens = data.get("prompt_cache_miss_tokens", 0)
    agent.last_prompt_tokens = data.get("last_prompt_tokens", 0)
    agent.delegation_count = data.get("delegation_count", 0)
    agent.token_budget = data.get("token_budget", 0)
    agent.session_id = data.get("id", session_id)
    agent.session_created = data.get("created", "")
    return agent


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------

def _redact_md(val: str) -> str:
    """Redact any API-key-looking substrings in *val*."""
    return _API_KEY_RX.sub("<REDACTED:API_KEY>", val)


def _truncate_result(text: str, max_chars: int = 500, max_lines: int = 10) -> str:
    """Return *text* truncated to *max_chars* chars and *max_lines* lines.

    If truncation is applied, ``... (truncated)`` is appended.
    """
    lines = text.splitlines()
    line_truncated = len(lines) > max_lines
    if line_truncated:
        lines = lines[:max_lines]
        text = "\n".join(lines)

    char_truncated = len(text) > max_chars
    if char_truncated:
        text = text[:max_chars]

    if line_truncated or char_truncated:
        text = text.rstrip("\n") + "\n... (truncated)"
    return text


def export_markdown(agent: "Agent", path: Path | str | None = None) -> Path:
    """Export the agent's conversation to a Markdown file.

    Parameters
    ----------
    agent:
        The agent whose messages to export.
    path:
        Output file path.  Defaults to ``agent.root / ".devbot" /
        f"{agent.session_id}.md"``.  If *agent* has no ``session_id``, one is
        generated via :func:`_make_session_id`.

    Returns
    -------
    Path
        The path that was written.
    """
    # Resolve path
    if path is None:
        sid = getattr(agent, "session_id", None)
        if sid is None:
            sid = _make_session_id()
            agent.session_id = sid
        path = _sessions_dir(agent.root) / f"{sid}.md"
    else:
        path = Path(path)

    # Determine mode
    if getattr(agent, "megaswarm", False):
        mode = "megaswarm"
    elif getattr(agent, "swarm", False):
        mode = "swarm"
    else:
        mode = "solo"

    cost = agent.estimated_cost()
    exported_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    # Build Markdown
    lines: list[str] = []
    lines.append("# DevBot Session Export")
    lines.append(f"**Model:** {agent.model}")
    lines.append(f"**Mode:** {mode}")
    lines.append(f"**Session tokens:** {agent.total_tokens:,}")
    lines.append(f"**Estimated cost:** ${cost:.2f}")
    lines.append(f"**Session ID:** {getattr(agent, 'session_id', 'N/A')}")
    lines.append(f"**Exported:** {exported_ts}")
    lines.append("")
    lines.append("---")
    lines.append("")

    messages = getattr(agent, "messages", []) or []

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Skip system messages
        if role == "system":
            i += 1
            continue

        if role == "user":
            lines.append("### User")
            lines.append(_redact_md(str(content)))
            lines.append("")

        elif role == "assistant":
            lines.append("### Assistant")
            # content may be None when only tool_calls are present
            if content:
                lines.append(_redact_md(str(content)))
                lines.append("")

            tool_calls = msg.get("tool_calls")
            if tool_calls:
                lines.append("**Tool calls:**")
                for tc in tool_calls:
                    # Redact args before rendering
                    fn = tc.get("function", {})
                    name = fn.get("name", "?")
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = raw_args
                    redacted = _redact_value_md(args)
                    block = json.dumps(
                        {"name": name, "arguments": redacted},
                        ensure_ascii=False,
                        indent=2,
                    )
                    lines.append("```json")
                    lines.append(block)
                    lines.append("```")
                    lines.append("")

            # Look ahead for tool-result messages that follow this assistant
            # message.  Each tool-result message has a tool_call_id that pairs
            # with one of the tool_calls above.
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                tmsg = messages[j]
                tool_name = tmsg.get("name", tmsg.get("tool_call_id", "?"))
                result = str(tmsg.get("content", ""))
                truncated = _truncate_result(_redact_md(result))
                lines.append(f"**Tool result** (`{tool_name}`):")
                lines.append("```")
                lines.append(truncated)
                lines.append("```")
                lines.append("")
                j += 1

            # Skip over the tool messages we just rendered
            i = j
            continue

        i += 1

    # Ensure directory exists and write
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _redact_value_md(val):
    """Recursively redact API-key-looking strings in *val* (dict/list/str).

    Same logic as devlog._redact_value but standalone to avoid a circular import.
    Uses ``.search()`` (not ``.match()``) so keys embedded in larger strings are
    caught.
    """
    if isinstance(val, str):
        return _API_KEY_RX.sub("<REDACTED:API_KEY>", val)
    if isinstance(val, dict):
        return {k: _redact_value_md(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_redact_value_md(v) for v in val]
    return val
