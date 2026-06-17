"""Tool implementations and JSON schemas exposed to the model."""

import ast
import difflib
import fnmatch
import locale
import os
import py_compile
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

MAX_OUTPUT = 50_000  # default chars returned to the model per tool call
DEFAULT_READ_FILE_LIMIT = 2000

_BACKUPS_DIR = ".devbot/backups"
_MAX_BACKUP_SETS = 20

# Directories skipped by recursive walks (grep, find_files) and listings.
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
              ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".next",
              "target", ".idea", ".vscode", ".egg-info", "__pypackages__",
              ".turbo", ".nx"}

# Dangerous shell patterns blocked in run_command.
BLOCKED_PATTERNS = [
    r"rm\s+-rf\s+/",
    r":\(\)\s*\{\s*:\|:&\s*\};:",
    r"chmod\s+.*777\s+/",
    r">\s*/dev/sda",
    r"mkfs\.",
    r"dd\s+if=",
    r"curl.*\|.*(sh|bash|dash|zsh|ksh)",
    r"wget.*\|.*(sh|bash|dash|zsh|ksh)",
    r"sudo\s",
    r"\.\./\.\.",
]

# Allow-list of safe/read-only command patterns for run_command.
ALLOW_LIST = [
    # ---- Git read-only ----
    r"^git\s+status(\s|$)",
    r"^git\s+diff(\s|$)",
    r"^git\s+log(\s|$)",
    r"^git\s+branch(\s|$)",
    r"^git\s+show(\s|$)",
    r"^git\s+stash\s+list(\s|$)",
    r"^git\s+remote\s+-v(\s|$)",
    # ---- Directory listing ----
    r"^(ls|dir|tree)(\s|$)",
    # ---- Python ----
    r"^pytest(\s|$)",
    r"^python3?\s",
    r"^pip\s+(list|freeze|show)(\s|$)",
    # ---- Node ----
    r"^npm\s+(test|run)(\s|$)",
    r"^npx(\s|$)",
    # ---- Cargo (fmt requires --check) ----
    r"^cargo\s+(test|check|build|clippy)(\s|$)",
    r"^cargo\s+fmt\s+.*--check",
    # ---- Go ----
    r"^go\s+(test|build|vet|fmt)(\s|$)",
    # ---- Basic read-only ----
    r"^(echo|cat|type|head|tail)(\s|$)",
    # ---- Other safe utilities ----
    r"^make\s+(test|check)(\s|$)",
    r"^(which|where)(\s|$)",
]


# Shell operators that allow chaining, redirection, or substitution. An
# allow-listed prefix containing any of these can't be trusted under shell=True.
_SHELL_META = re.compile(r"[;&|`><\n]|\$\(")


def _glob_match(path_str: str, pattern: str) -> bool:
    """Match a relative path (forward-slash) against a glob pattern.

    If *pattern* contains no ``/``, matching is done against the bare
    filename only (the last component of *path_str*).  This preserves
    backward compatibility with ``Path(name).match(glob)``.

    If *pattern* contains ``/``, the path and pattern are split on ``/``
    and matched segment-by-segment, with ``**`` matching zero or more
    path segments.
    """
    # Normalize backslashes → forward slashes so Windows paths work.
    path_str = path_str.replace("\\", "/")
    pattern = pattern.replace("\\", "/")

    if "/" not in pattern:
        # Bare filename pattern — match against the last component only.
        basename = path_str.rsplit("/", 1)[-1]
        return fnmatch.fnmatch(basename, pattern)

    path_parts = path_str.split("/")
    pat_parts = pattern.split("/")

    def _match_segments(pi: int, pj: int) -> bool:
        """Match pat_parts[pi:] against path_parts[pj:]."""
        while pi < len(pat_parts) and pj < len(path_parts):
            if pat_parts[pi] == "**":
                if pi == len(pat_parts) - 1:
                    # Trailing ** matches everything (including empty).
                    return True
                # Mid-path **: try matching zero or more path segments.
                for skip in range(len(path_parts) - pj + 1):
                    if _match_segments(pi + 1, pj + skip):
                        return True
                return False
            else:
                if not fnmatch.fnmatch(path_parts[pj], pat_parts[pi]):
                    return False
                pi += 1
                pj += 1

        # Consumed all pattern parts — need all path parts consumed too.
        if pi == len(pat_parts):
            return pj == len(path_parts)
        # Consumed all path parts — remaining pattern must be only **.
        if pj == len(path_parts):
            return pi == len(pat_parts) - 1 and pat_parts[pi] == "**"

        return False

    return _match_segments(0, 0)


def _parse_gitignore(root: Path) -> tuple[list[str], list[str]]:
    """Parse ``root / ".gitignore"`` and return (file_patterns, dir_patterns).

    * Blank lines and lines starting with ``#`` are ignored.
    * Trailing whitespace and trailing ``/`` are stripped from each pattern.
    * A pattern that originally ended with ``/`` is a directory-only pattern
      (e.g. ``__pycache__/`` → dir_patterns).  Everything else is a file
      pattern (e.g. ``*.pyc`` → file_patterns).
    * Returns ([], []) when the file does not exist or is empty.
    """
    gf = root / ".gitignore"
    if not gf.is_file():
        return [], []

    file_patterns: list[str] = []
    dir_patterns: list[str] = []
    for raw in gf.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.rstrip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("/"):
            dir_patterns.append(line[:-1])
        else:
            file_patterns.append(line)
    return file_patterns, dir_patterns


def _is_gitignored(rel_path: str, file_patterns: list[str], dir_patterns: list[str]) -> bool:
    """Return True if *rel_path* (forward-slash normalised) matches any gitignore pattern.

    * *dir_patterns* — a path component exactly equals the pattern → ignored.
    * *file_patterns* — ``_glob_match(rel_path, pattern)`` → ignored.
    """
    # Directory patterns: any path component equals the pattern.
    for comp in rel_path.split("/"):
        for dp in dir_patterns:
            if comp == dp:
                return True

    # File patterns: use the existing glob engine.
    for fp in file_patterns:
        if _glob_match(rel_path, fp):
            return True

    return False


class PathError(Exception):
    """Raised when a path resolves outside the sandboxed project root."""


def check_command(command: str) -> tuple:
    """Check a command against blocked and allow-list patterns.

    Returns a tuple ``(status: str, reason: str)`` where *status* is one of:

    * ``"blocked"`` – matches a dangerous (deny-list) pattern
    * ``"allowed"`` – matches a safe (allow-list) pattern
    * ``"needs_approval"`` – matches neither list
    """
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command):
            return ("blocked", f"matches dangerous pattern '{pattern}'")
    for pattern in ALLOW_LIST:
        if re.search(pattern, command):
            # The allow-list only validates the command *prefix*. Under
            # shell=True, operators like ; & | $() `` > < let an allow-listed
            # prefix chain into arbitrary commands (e.g. "git status && rm ...").
            # If any shell metacharacter is present, fall through to approval.
            if _SHELL_META.search(command):
                return ("needs_approval",
                        "allow-listed command contains shell metacharacters "
                        "(chaining/redirection) — requires approval")
            return ("allowed", f"matches allow-list pattern '{pattern}'")
    return ("needs_approval", "not on allow-list")


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    """Read a non-negative int from env, falling back on bad values."""
    try:
        value = int(os.environ.get(name, str(default)) or str(default))
    except ValueError:
        return default
    return max(minimum, value)


def _max_output() -> int:
    return _env_int("DEVBOT_MAX_TOOL_OUTPUT", MAX_OUTPUT, minimum=1000)


def _default_read_limit() -> int:
    return _env_int("DEVBOT_READ_FILE_LIMIT", DEFAULT_READ_FILE_LIMIT, minimum=1)


def _clip(text: str) -> str:
    max_output = _max_output()
    if len(text) >= max_output:
        return text[:max_output] + f"\n... [truncated, {len(text)} chars total]"
    return text


DIFF_CLIP = 2000   # default chars of unified diff returned to the model


def _compute_diff(old_content: str, new_content: str, filename: str) -> str:
    """Return a fenced ```diff block showing the unified diff, or a minimal note."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff_lines = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filename}", tofile=f"b/{filename}",
    ))

    if not diff_lines:
        return "(no changes)"

    diff_text = "".join(diff_lines)
    full_block = f"```diff\n{diff_text}```"

    diff_clip = _env_int("DEVBOT_DIFF_CLIP", DIFF_CLIP, minimum=200)
    if len(full_block) > diff_clip:
        full_block = full_block[:diff_clip] + "\n... [diff truncated]"

    return full_block


def _resolve(path: str, root: Path) -> Path:
    """Resolve path against root and ensure it stays inside root.

    Both relative and absolute paths are allowed, but the resolved location
    must be within the project root — this sandboxes the agent so it cannot
    read or write arbitrary files elsewhere on the machine.
    """
    root = root.resolve()
    p = Path(path)
    p = (root / p).resolve() if not p.is_absolute() else p.resolve()
    # Allow access to root itself and anything inside it
    if p != root and not p.is_relative_to(root):
        raise PathError(
            f"Path '{path}' is outside the project root ({root}). "
            "Access is restricted to the project directory."
        )
    return p


def read_file(path: str, root: Path, offset: int = 0,
              limit: int | None = None) -> str:
    p = _resolve(path, root)
    if not p.is_file():
        return f"Error: {p} is not a file"
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(lines)
    offset = max(0, offset)
    limit = _default_read_limit() if limit is None else max(1, limit)
    if offset >= total:
        return f"Error: offset {offset} is beyond file end ({total} lines)"
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


def _backup_file(root: Path, file_path: Path, old_content: str) -> None:
    """Save a backup of *file_path* before it is modified."""
    ts = str(time.time_ns())
    backup_dir = root / _BACKUPS_DIR / ts
    rel = file_path.relative_to(root)
    (backup_dir / rel).parent.mkdir(parents=True, exist_ok=True)
    (backup_dir / rel).write_text(old_content, encoding="utf-8")
    _prune_backups(root)


def _prune_backups(root: Path) -> None:
    """Keep only the most recent _MAX_BACKUP_SETS backup folders."""
    backups_root = root / _BACKUPS_DIR
    if not backups_root.is_dir():
        return
    dirs = sorted(
        [d for d in backups_root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    for old in dirs[_MAX_BACKUP_SETS:]:
        shutil.rmtree(old)


def undo_last_edit(root: Path) -> str:
    """Restore files from the most recent backup set and remove it."""
    backups_root = root / _BACKUPS_DIR
    if not backups_root.is_dir():
        return "No backups found."
    dirs = sorted(
        [d for d in backups_root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    if not dirs:
        return "No backups found."
    latest = dirs[0]
    restored = []
    for f in sorted(latest.rglob("*")):
        if f.is_file():
            rel = f.relative_to(latest)
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
            restored.append(str(rel))
    shutil.rmtree(latest)
    if not restored:
        return "Backup set was empty; nothing to restore."
    lines = [f"Restored {len(restored)} file(s) from backup:"]
    for r in restored:
        lines.append(f"  {r}")
    return "\n".join(lines)


def write_file(path: str, content: str, root: Path) -> str:
    p = _resolve(path, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.is_file()
    old_content = p.read_text(encoding="utf-8", errors="replace") if existed else None
    if existed:
        _backup_file(root, p, old_content)
    p.write_text(content, encoding="utf-8")
    msg = f"Wrote {len(content)} chars to {p}"
    if existed:
        diff = _compute_diff(old_content, content, str(p))
        msg += "\n" + diff
    else:
        lines = content.count("\n") + (0 if content.endswith("\n") else 1)
        msg += f"\n({lines} lines, new file)"
    return msg


def edit_file(path: str, old_string: str, new_string: str, root: Path,
              replace_all: bool = False) -> str:
    if not old_string:
        return "Error: old_string must not be empty."
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
    new_text = text.replace(old_string, new_string, -1 if replace_all else 1)
    _backup_file(root, p, text)
    p.write_text(new_text, encoding="utf-8")
    diff = _compute_diff(text, new_text, str(p))
    return f"Replaced {count if replace_all else 1} occurrence(s) in {p}\n" + diff


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


def grep(pattern: str, root: Path, path: str = ".", glob: str = "",
         respect_gitignore: bool = True) -> str:
    base = _resolve(path, root)
    rx = re.compile(pattern)
    file_pats, dir_pats = ([], [])
    if respect_gitignore:
        file_pats, dir_pats = _parse_gitignore(root)
    hits = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            rel = str((Path(dirpath) / name).relative_to(base)).replace("\\", "/")
            if glob:
                if not _glob_match(rel, glob):
                    continue
            if respect_gitignore and _is_gitignored(rel, file_pats, dir_pats):
                continue
            fp = Path(dirpath) / name
            try:
                for n, line in enumerate(fp.read_text(encoding="utf-8").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{rel}:{n}: {line.strip()}")
                        if len(hits) >= 200:
                            return _clip("\n".join(hits) + "\n... [stopped at 200 matches]")
            except (UnicodeDecodeError, PermissionError, OSError):
                continue
    return _clip("\n".join(hits) or "No matches")


def find_files(glob: str, root: Path, path: str = ".",
               respect_gitignore: bool = True) -> str:
    base = _resolve(path, root)
    file_pats, dir_pats = ([], [])
    if respect_gitignore:
        file_pats, dir_pats = _parse_gitignore(root)
    matches = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            rel = str((Path(dirpath) / name).relative_to(base)).replace("\\", "/")
            if _glob_match(rel, glob):
                if respect_gitignore and _is_gitignored(rel, file_pats, dir_pats):
                    continue
                matches.append(rel)
                if len(matches) >= 500:
                    return _clip("\n".join(matches) + "\n... [stopped at 500 files]")
    return _clip("\n".join(sorted(matches)) or "No files matched")


def run_command(command: str, root: Path, timeout: int = 120) -> str:
    # Shell injection hardening: block dangerous patterns.
    status, reason = check_command(command)
    if status == "blocked":
        return f"Error: command blocked — {reason}"
    # Detect the system encoding with a utf-8 fallback (fixes Windows output).
    enc = sys.stdout.encoding or locale.getpreferredencoding() or "utf-8"
    try:
        result = subprocess.run(
            command, shell=True, cwd=root, capture_output=True,
            text=True, timeout=timeout, encoding=enc, errors="replace",
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    out = result.stdout or ""
    if result.stderr.strip():
        out += ("\n[stderr]\n" + result.stderr)
    out += f"\n[exit code: {result.returncode}]"
    return _clip(out.strip())


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return structured results.

    Requires the `ddgs` package (>=9.0). Falls back to the DuckDuckGo
    Instant Answer API if the package is not installed, though that only
    provides encyclopedic summaries, not full web results.
    """
    max_results = min(max(max_results, 1), 10)

    fallback = False
    try:
        from ddgs import DDGS

        results = list(DDGS().text(query, max_results=max_results))
    except ImportError:
        fallback = True
    except Exception as e:
        return f"Error: web search via ddgs failed — {e}"

    if fallback:
        # Fallback to the free Instant Answer API (no API key needed).
        import json
        import urllib.request
        import urllib.parse

        url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({
            "q": query, "format": "json", "no_html": "1", "skip_disambig": "1",
        })
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            return f"Error: web search failed — {e}"

        lines = []
        abstract = (data.get("AbstractText") or "").strip()
        if abstract:
            source = data.get("AbstractURL") or ""
            heading = data.get("Heading") or "Result"
            lines.append(f"[{heading}]")
            lines.append(f"   {abstract}")
            if source:
                lines.append(f"   URL: {source}")
        else:
            lines.append("(no instant answer found for this query)")

        related = data.get("RelatedTopics") or []
        for i, topic in enumerate(related[:max_results]):
            text = (topic.get("Text") or "").strip()
            url = topic.get("FirstURL") or ""
            if text:
                lines.append(f"\n{i + 1}. {text}")
                if url:
                    lines.append(f"   URL: {url}")
        return _clip("\n".join(lines))

    if not results:
        return "(no web results found for this query)"

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "(no title)")
        href = r.get("href", "")
        body = r.get("body", "")
        lines.append(f"{i}. {title}")
        if href:
            lines.append(f"   URL: {href}")
        if body:
            lines.append(f"   {body}")
        lines.append("")
    return _clip("\n".join(lines))


def verify(root: Path) -> str:
    """Auto-detect the project's test framework, run tests, and report results.

    Checks for common test runners (pytest, tox, npm test, cargo test, go test,
    make test) and runs the first one that looks viable. Falls back to a manual
    check of recently modified Python files for syntax errors.
    """
    # Ordered list of (condition, command, label) triples
    candidates: list[tuple] = [
        # Python
        ((root / "pyproject.toml").is_file() or (root / "setup.py").is_file()
         or (root / "setup.cfg").is_file(),
         ["pytest", "-x", "--tb=short", "-q"], "pytest"),

        ((root / "tox.ini").is_file(), ["tox", "-q"], "tox"),

        # Node/JS
        ((root / "package.json").is_file(),
         ["npm", "test", "--", "--silent"], "npm test"),

        # Rust
        ((root / "Cargo.toml").is_file(), ["cargo", "test", "-q"], "cargo test"),

        # Go
        ((root / "go.mod").is_file(), ["go", "test", "./..."], "go test"),

        # Make
        ((root / "Makefile").is_file(), ["make", "test"], "make test"),
    ]

    for condition, cmd, label in candidates:

        if not condition:
            continue

        try:
            result = subprocess.run(
                cmd, cwd=root, capture_output=True,
                text=True, timeout=180, encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired:
            return f"WARN: {label} timed out after 180s"
        except FileNotFoundError:
            continue  # runner not installed, try next

        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        rc = result.returncode

        if rc == 5:
            # "no tests collected" — report and stop trying other candidates
            return (f"INFO: pytest ran but collected no tests (exit 5). "
                    "Check test discovery config.")

        report = [f"[TEST] {label} (exit {rc})"]
        if out:
            report.append("\n[stdout]")
            report.append(_clip(out))
        if err:
            report.append("\n[stderr]")
            report.append(_clip(err))
        if rc == 0:
            report.insert(0, "PASS: All tests passed!")
        else:
            report.insert(0, f"FAIL: Tests failed (exit code {rc})")
        return "\n".join(report)

    # No test framework found — do a quick syntax check on Python files
    py_files = list(root.rglob("*.py"))
    if not py_files:
        return "WARN: No test framework detected and no Python files to check."

    errors = []
    for fp in py_files:
        # Skip common virtualenv / cache paths
        parts = fp.parts
        if any(d in SKIP_DIRS for d in parts):
            continue
        try:
            py_compile.compile(str(fp), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(f"  Syntax error in {fp.relative_to(root)}:\n    {e}")
        except Exception:
            pass

    if errors:
        return "WARN: No test framework found. Syntax check found issues:\n" + "\n".join(errors[:20])
    checked = len(py_files) - sum(1 for fp in py_files if any(d in SKIP_DIRS for d in fp.parts))
    return f"INFO: No test framework detected. Syntax-checked {checked} Python files -- no errors."


# ---- outline helper: regex for non-Python definition-like lines ----
_OUTLINE_RE = re.compile(
    r"^\s*(def|class|function|fn|func|public|private|protected|export)\b"
)


def _build_func_sig(node: ast.AST) -> str:
    """Build a parameter signature string from an AST function node.

    Uses ``ast.unparse`` for default values (Python ≥3.9).  Falls back to
    ``repr`` when ``ast.unparse`` is not available.
    """
    args = node.args
    parts: list[str] = []

    _unparse = getattr(ast, "unparse", None)

    def _dump(val):
        if _unparse is not None:
            try:
                return _unparse(val)
            except Exception:
                pass
        return repr(val)

    # Positional-only args (Python ≥3.8)
    posonly = getattr(args, "posonlyargs", None) or []
    # defaults are right-aligned against the combined posonly + args list
    total_positional = len(posonly) + len(args.args)
    num_no_default = total_positional - len(args.defaults)
    for i, arg in enumerate(posonly):
        if i >= num_no_default:
            d = _dump(args.defaults[i - num_no_default])
            parts.append(f"{arg.arg}={d}")
        else:
            parts.append(arg.arg)
    if posonly:
        parts.append("/")

    # Positional-or-keyword args
    for i, arg in enumerate(args.args):
        global_i = len(posonly) + i
        if global_i >= num_no_default:
            d = _dump(args.defaults[global_i - num_no_default])
            parts.append(f"{arg.arg}={d}")
        else:
            parts.append(arg.arg)

    # *args
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        # bare * separator when there are keyword-only args but no *args
        parts.append("*")

    # Keyword-only args
    kw_defaults = args.kw_defaults if args.kw_defaults else []
    for i, arg in enumerate(args.kwonlyargs):
        if i < len(kw_defaults) and kw_defaults[i] is not None:
            d = _dump(kw_defaults[i])
            parts.append(f"{arg.arg}={d}")
        else:
            parts.append(arg.arg)

    # **kwargs
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    return "(" + ", ".join(parts) + ")"


def _walk_defs(body: list[ast.AST], indent: int = 0) -> list[str]:
    """Recursively walk AST body, yielding outline lines with 2-space nesting."""
    lines: list[str] = []
    prefix = " " + "  " * indent  # " " + 2 spaces per nesting level
    for node in body:
        if isinstance(node, ast.FunctionDef):
            sig = _build_func_sig(node)
            lines.append(f"{node.lineno}:{prefix}def {node.name}{sig}")
        elif isinstance(node, ast.AsyncFunctionDef):
            sig = _build_func_sig(node)
            lines.append(f"{node.lineno}:{prefix}async def {node.name}{sig}")
        elif isinstance(node, ast.ClassDef):
            bases = ""
            if node.bases:
                try:
                    bases = "(" + ", ".join(ast.unparse(b) for b in node.bases) + ")"
                except Exception:
                    bases = "(" + ", ".join(repr(b) for b in node.bases) + ")"
            lines.append(f"{node.lineno}:{prefix}class {node.name}{bases}")
            lines.extend(_walk_defs(node.body, indent + 1))
    return lines


def _outline_python(p: Path) -> str:
    """Parse a Python file with ``ast`` and return a nested definition outline."""
    try:
        source = p.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"Error: syntax error in {p}: {e.msg}"
    except Exception as e:
        return f"Error: {e}"

    lines = _walk_defs(tree.body)
    return "\n".join(lines) if lines else "(no definitions found)"


def _outline_regex(p: Path) -> str:
    """Fallback outline for non-Python files using a regex over source lines."""
    try:
        text_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return f"Error: {e}"

    result = []
    for i, line in enumerate(text_lines, 1):
        if _OUTLINE_RE.match(line):
            result.append(f"{i}: {line.strip()}")
    return "\n".join(result) if result else "(no definitions found)"


def tree(path: str, root: Path, max_depth: int = 3) -> str:
    """Render an indented directory tree starting at *path*.

    Uses Unicode box-drawing characters (├── └── │).  The starting directory
    is shown as a header line (depth 0); its direct children are depth 1, and
    so on up to *max_depth* levels of nesting below the start.

    * Directories listed in ``SKIP_DIRS`` are pruned during traversal.
    * ``.gitignore`` patterns (parsed from *root*) are respected — relative
      paths are computed from *root* for matching.
    * Total entries (files + directories) is capped at 500 with a truncation
      note.
    * If *path* is not a directory, an error string is returned.
    * The final output is passed through ``_clip``.
    """
    p = _resolve(path, root)
    if not p.is_dir():
        return f"Error: {p} is not a directory"

    file_pats, dir_pats = _parse_gitignore(root)

    lines: list[str] = [str(p)]
    entry_count: list[int] = [0]   # mutable so nested _walk can mutate
    truncated: list[bool] = [False]

    def _walk(dir_path: Path, depth: int, prefix: str) -> None:
        if depth > max_depth or truncated[0]:
            return

        try:
            entries = sorted(dir_path.iterdir(),
                             key=lambda x: (x.is_dir(), x.name.lower()))
        except (PermissionError, OSError):
            return

        # Filter out SKIP_DIRS and gitignored entries.
        filtered: list[Path] = []
        for e in entries:
            if e.name in SKIP_DIRS:
                continue
            # Compute path relative to *root* for gitignore checks.
            try:
                rel = str(e.relative_to(root)).replace("\\", "/")
            except ValueError:
                rel = str(e)
            if _is_gitignored(rel, file_pats, dir_pats):
                continue
            filtered.append(e)

        for i, e in enumerate(filtered):
            if entry_count[0] >= 500:
                truncated[0] = True
                return

            is_last = (i == len(filtered) - 1)
            connector = "└── " if is_last else "├── "
            entry_count[0] += 1
            suffix = "/" if e.is_dir() else ""
            lines.append(prefix + connector + e.name + suffix)

            if e.is_dir():
                child_prefix = prefix + ("    " if is_last else "│   ")
                _walk(e, depth + 1, child_prefix)

    _walk(p, 1, "")

    if truncated[0]:
        lines.append("... [truncated at 500 entries]")

    return _clip("\n".join(lines))


def outline(path: str, root: Path) -> str:
    """Return an outline of top-level and nested class/function definitions in a file.

    Python files (``.py``) are parsed with the stdlib ``ast`` module, yielding
    each definition with its line number, nesting-aware indentation, and
    signature.  Non-Python files fall back to a regex that matches lines
    starting with ``def``, ``class``, ``function``, ``fn``, ``func``,
    ``public``, ``private``, ``protected``, or ``export``.
    """
    p = _resolve(path, root)
    if not p.is_file():
        if p.is_dir():
            return f"Error: {p} is a directory, not a file"
        return f"Error: {p} is not a file"

    if p.suffix.lower() == ".py":
        return _outline_python(p)
    else:
        return _outline_regex(p)


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
            "name": "tree",
            "description": "Render an indented directory tree. Prunes hidden/skipped directories and gitignored entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                    "max_depth": {"type": "integer", "description": "Maximum depth of nesting to show (default 3)", "default": 3},
                },
                "required": ["path"],
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
                    "glob": {"type": "string", "description": "Filename glob filter, e.g. *.py or src/**/*.py. Supports ** for recursive matching."},
                    "respect_gitignore": {"type": "boolean", "description": "When True (default), skip files matching .gitignore patterns in the project root."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "Find files by name using a glob pattern, e.g. *.py or test_*.txt. Searches recursively. Supports ** patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "glob": {"type": "string", "description": "Filename glob, e.g. *.py or **/*.py. Supports ** for recursive matching."},
                    "path": {"type": "string", "description": "Directory to search, default project root"},
                    "respect_gitignore": {"type": "boolean", "description": "When True (default), skip files matching .gitignore patterns in the project root."},
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
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo and return structured results (titles, URLs, snippets). Use for finding documentation, examples, or current info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query string."},
                    "max_results": {"type": "integer", "description": "Maximum number of results to return (default 5, max 10).", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "outline",
            "description": "Return an outline of top-level and nested class/function definitions in a file. Python files are parsed with AST; non-Python files use a definition regex.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (relative to project root or absolute)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify",
            "description": "Auto-detect the project's test framework (pytest, npm test, cargo test, go test, etc.), run tests, and report results. Use after making changes to verify they work.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "undo_last_edit",
            "description": "Restore files from the most recent edit backup. Reverts the last write_file or edit_file operation.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# Tools that mutate state or execute code -> require confirmation unless auto-approved
DANGEROUS_TOOLS = {"write_file", "edit_file", "run_command", "verify", "undo_last_edit"}

# Dict-based dispatch: maps tool name -> handler(args, root).
# Null-safe arg.get(key) or default handles JSON null coercion (P1-11).
_TOOL_HANDLERS = {
    "read_file": lambda a, r: read_file(
        a["path"], r, a.get("offset") or 0, a.get("limit")),
    "write_file": lambda a, r: write_file(a["path"], a["content"], r),
    "edit_file": lambda a, r: edit_file(
        a["path"], a["old_string"], a["new_string"], r, a.get("replace_all") or False),
    "list_dir": lambda a, r: list_dir(a.get("path") or ".", r),
    "tree": lambda a, r: tree(a["path"], r, a.get("max_depth") if a.get("max_depth") is not None else 3),
    "grep": lambda a, r: grep(
        a["pattern"], r, a.get("path") or ".", a.get("glob") or "",
        a.get("respect_gitignore", True)),
    "find_files": lambda a, r: find_files(
        a["glob"], r, a.get("path") or ".",
        a.get("respect_gitignore", True)),
    "run_command": lambda a, r: run_command(
        a["command"], r, a.get("timeout") or 120),
    "web_search": lambda a, r: web_search(
        a["query"], a.get("max_results") or 5),
    "verify": lambda a, r: verify(r),
    "outline": lambda a, r: outline(a["path"], r),
    "undo_last_edit": lambda a, r: undo_last_edit(r),
}


def dispatch(name: str, args: dict, root: Path) -> str:
    """Execute a tool by name with parsed arguments."""
    try:
        handler = _TOOL_HANDLERS.get(name)
        if handler is None:
            return f"Error: unknown tool '{name}'"
        return handler(args, root)
    except Exception as e:  # surface errors to the model so it can recover
        return f"Error: {type(e).__name__}: {e}"
