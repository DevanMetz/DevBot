"""JSONL logging helper for DevBot sessions.

When DEVBOT_LOG is set to a file path, appends one JSON object per line (JSONL)
for tool calls and assistant turns.  When unset, all functions are no-ops.

API-key redaction: any string argument whose value starts with ``sk-`` followed
by alphanumerics is replaced with ``<REDACTED:API_KEY>``.
"""

import json
import os
import re
import time
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Cached env-var lookup (once per process)
# ---------------------------------------------------------------------------
_LOGFILE: str | None = os.environ.get("DEVBOT_LOG") or None

# Pattern for API keys we must never log.
_API_KEY_RX = re.compile(r"^sk-[a-zA-Z0-9]+$")


def _redact_value(val: Any, seen: set | None = None) -> Any:
    """Recursively redact API-key-looking strings in *val* (dict/list/str)."""
    if seen is None:
        seen = set()

    if isinstance(val, str):
        return "<REDACTED:API_KEY>" if _API_KEY_RX.match(val) else val

    if isinstance(val, dict):
        obj_id = id(val)
        if obj_id in seen:
            return val  # cycle guard
        seen.add(obj_id)
        return {k: _redact_value(v, seen) for k, v in val.items()}

    if isinstance(val, list):
        obj_id = id(val)
        if obj_id in seen:
            return val
        seen.add(obj_id)
        return [_redact_value(v, seen) for v in val]

    return val


def _args_summary(args: dict) -> dict:
    """Return a shallow copy of *args* with values replaced by repr-truncated
    snippets and API keys redacted.  Safe to serialise."""
    summary: Dict[str, Any] = {}
    for k, v in args.items():
        v = _redact_value(v)
        if isinstance(v, str):
            s = v
            if len(s) > 120:
                s = s[:120] + "..."
            summary[k] = s
        elif isinstance(v, (int, float, bool)) or v is None:
            summary[k] = v
        else:
            # e.g. nested dict — still redact but coerce to a compact repr
            s = repr(v)
            if len(s) > 120:
                s = s[:120] + "..."
            summary[k] = s
    return summary


def _append(record: dict) -> None:
    """Append one JSON line to the log file.  Errors are silently ignored."""
    if not _LOGFILE:
        return
    try:
        with open(_LOGFILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            fh.write("\n")
    except Exception:
        pass  # logging must NEVER crash the agent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_tool_call(name: str, args: dict, result: str, ok: bool) -> None:
    """Log a tool invocation."""
    if not _LOGFILE:
        return
    record = {
        "type": "tool_call",
        "timestamp": time.time(),
        "name": name,
        "args": _args_summary(args),
        "result_length": len(result),
        "ok": ok,
    }
    _append(record)


def log_turn(model: str, prompt_tokens: int, total_tokens: int) -> None:
    """Log an assistant turn."""
    if not _LOGFILE:
        return
    record = {
        "type": "assistant_turn",
        "timestamp": time.time(),
        "model": model,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
    }
    _append(record)
