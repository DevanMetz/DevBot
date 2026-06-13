"""Tool implementations and JSON schemas exposed to the model."""

import os
import re
import subprocess
from pathlib import Path

MAX_OUTPUT = 50_000  # chars returned to the model per tool call

# Directories skipped by recursive walks (grep, find_files) and listings.
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}


class PathError(Exception):
    """Raised when a path resolves outside the sandboxed project root."""


def _clip(text: str) -> str:
    if len(text) > MAX_OUTPUT:
        return text[:MAX_OUTPUT] + f"\n... [truncated, {len(text)} chars total]"
    return text


def _resolve(path: str, root: Path) -> Path:
    """Resolve path against root and ensure it stays inside root.

    Both relative and absolute paths are allowed, but the resolved location
    must be within the project root — this sandboxes the agent so it cannot
    read or write arbitrary files elsewhere on the machine.
    """
    root = root.resolve()
    p = Path(path)
    p = (root / p).resolve() if not p.is_absolute() else p.resolve()
    if p != root and root not in p.parents:
        raise PathError(
            f"Path '{path}' is outside the project root ({root}). "
            "Access is restricted to the project directory."
        )
    return p


def read_file(path: str, root: Path, offset: int = 0, limit: int = 2000) -> str:
    p = _resolve(path, root)
    if not p.is_file():
        return f"Error: {p} is not a file"
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(lines)
    chunk = lines[offset : offset + limit]
    numbered = "\n".join(f"{i + offset + 1}\t{line}" for i, line in enumerate(chunk))
    if not numbered:
        return "(empty file)"
    # Tell the model when there is more content beyond this slice.
    end = offset + len(chunk)
    if offset > 0 or end < total:
        header = f"[showing lines {offset + 1}-{end} of {total}; use offset/limit for more]\n"
        return _clip(header + numbered)
    return _clip(numbered)


def write_file(path: str, content: str, root: Path) -> str:
    p = _resolve(path, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {p}"


def edit_file(path: str, old_string: str, new_string: str, root: Path,
              replace_all: bool = False) -> str:
    p = _resolve(path, root)
    if not p.is_file():
        return f"Error: {p} is not a file"
    text = p.read_text(encoding="utf-8")
    count = text.count(old_string)
    if count == 0:
        # Help the model self-correct: point at the closest line by first line of old_string.
        first = old_string.lstrip().splitlines()[0] if old_string.strip() else ""
        hint = ""
        if first:
            for n, line in enumerate(text.splitlines(), 1):
                if first in line:
                    hint = f" Closest line is {n}: {line.strip()[:120]!r}"
                    break
        return "Error: old_string not found in file (check exact whitespace/indentation)." + hint
    if count > 1 and not replace_all:
        return f"Error: old_string occurs {count} times; make it unique or set replace_all"
    text = text.replace(old_string, new_string, -1 if replace_all else 1)
    p.write_text(text, encoding="utf-8")
    return f"Replaced {count if replace_all else 1} occurrence(s) in {p}"


def list_dir(path: str, root: Path) -> str:
    p = _resolve(path or ".", root)
    if not p.is_dir():
        return f"Error: {p} is not a directory"
    entries = []
    for e in sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
        if e.name in SKIP_DIRS:
            continue
        entries.append(f"{e.name}/" if e.is_dir() else e.name)
    return _clip("\n".join(entries) or "(empty)")


def grep(pattern: str, root: Path, path: str = ".", glob: str = "") -> str:
    base = _resolve(path, root)
    rx = re.compile(pattern)
    hits = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if glob and not Path(name).match(glob):
                continue
            fp = Path(dirpath) / name
            try:
                for n, line in enumerate(fp.read_text(encoding="utf-8").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{fp.relative_to(base)}:{n}: {line.strip()}")
                        if len(hits) >= 200:
                            return _clip("\n".join(hits) + "\n... [stopped at 200 matches]")
            except (UnicodeDecodeError, PermissionError, OSError):
                continue
    return _clip("\n".join(hits) or "No matches")


def find_files(glob: str, root: Path, path: str = ".") -> str:
    base = _resolve(path, root)
    matches = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if Path(name).match(glob):
                matches.append(str((Path(dirpath) / name).relative_to(base)))
                if len(matches) >= 500:
                    return _clip("\n".join(matches) + "\n... [stopped at 500 files]")
    return _clip("\n".join(sorted(matches)) or "No files matched")


def run_command(command: str, root: Path, timeout: int = 120) -> str:
    try:
        result = subprocess.run(
            command, shell=True, cwd=root, capture_output=True,
            text=True, timeout=timeout, encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    out = result.stdout or ""
    if result.stderr:
        out += ("\n[stderr]\n" + result.stderr)
    out += f"\n[exit code: {result.returncode}]"
    return _clip(out.strip())


# JSON schemas sent to DeepSeek (OpenAI function-calling format)
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file, returning numbered lines. Use offset/limit for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (relative to project root or absolute)"},
                    "offset": {"type": "integer", "description": "Line offset to start from (0-based)"},
                    "limit": {"type": "integer", "description": "Max lines to read (default 2000)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the given content. Creates parent directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact string in a file. old_string must match exactly (including whitespace) and be unique unless replace_all is true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories at a path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory path, default project root"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents recursively with a Python regex. Returns file:line: matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "Directory to search, default project root"},
                    "glob": {"type": "string", "description": "Filename glob filter, e.g. *.py"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "Find files by name using a glob pattern, e.g. *.py or test_*.txt. Searches recursively.",
            "parameters": {
                "type": "object",
                "properties": {
                    "glob": {"type": "string", "description": "Filename glob, e.g. *.py"},
                    "path": {"type": "string", "description": "Directory to search, default project root"},
                },
                "required": ["glob"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the project root and return stdout/stderr. Requires user approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "description": "Seconds, default 120"},
                },
                "required": ["command"],
            },
        },
    },
]

# Tools that mutate state or execute code -> require confirmation unless auto-approved
DANGEROUS_TOOLS = {"write_file", "edit_file", "run_command"}


def dispatch(name: str, args: dict, root: Path) -> str:
    """Execute a tool by name with parsed arguments."""
    try:
        if name == "read_file":
            return read_file(args["path"], root, args.get("offset", 0), args.get("limit", 2000))
        if name == "write_file":
            return write_file(args["path"], args["content"], root)
        if name == "edit_file":
            return edit_file(args["path"], args["old_string"], args["new_string"],
                             root, args.get("replace_all", False))
        if name == "list_dir":
            return list_dir(args.get("path", "."), root)
        if name == "grep":
            return grep(args["pattern"], root, args.get("path", "."), args.get("glob", ""))
        if name == "find_files":
            return find_files(args["glob"], root, args.get("path", "."))
        if name == "run_command":
            return run_command(args["command"], root, args.get("timeout", 120))
        return f"Error: unknown tool '{name}'"
    except Exception as e:  # surface errors to the model so it can recover
        return f"Error: {type(e).__name__}: {e}"
