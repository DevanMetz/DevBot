# DevBot improvement plan (autopilot-ready)

Run with `devbot --run-plan plan.md`. Each `## Phase` is independent and
self-contained. Every phase MUST add or update tests under `tests/` and leave
`pytest -q` green (CI runs it). Use the `pipeline` tool for all code changes;
never bare write/edit as the manager. No live API calls in tests — use
`tmp_path`/`monkeypatch`, mock agents. Preserve the path sandbox and the
approval-safety invariant. If a phase adds a `DEVBOT_*` env var, document it in
the README config table (test_phase5 enforces README↔code parity).

These are genuinely new items. Already done (do NOT redo): test suite + CI,
shell sandbox, sessions + resume + markdown export, cost controls + cache-hit
costing, megaswarm pipeline/divide, dashboard, reasoning suppression,
web_search, verify, recursive glob, JSONL logging, readline UX + tab
completion, global/loop token guards, /tools, /cost.

---

## Phase 1 — Code outline tool

Reading a whole file just to find a function wastes tokens. Add a fast,
read-only `outline` tool that summarizes a source file's structure.

**What to do:**
- Add `outline(path, root)` in `devbot/tools.py`. For Python files, parse with
  the stdlib `ast` module and return each top-level and nested class/function
  with its line number and signature (name + args), indented by nesting.
- For non-Python text files, fall back to listing lines that look like
  definitions (e.g. lines matching `^\s*(def|class|function|fn|func|public|private)\b`).
- Register it in `TOOL_SCHEMAS` and `_TOOL_HANDLERS`; it is read-only (no
  approval needed) and must respect the path sandbox.

**Done when:** `outline` on a Python file lists its classes/functions with line
numbers; a syntax-error file returns a clear error (not a crash); new tests
cover both. `pytest -q` green.

---

## Phase 2 — Project tree tool

There's no quick way to see project structure. Add a read-only `tree` tool.

**What to do:**
- Add `tree(path, root, max_depth=3)` in `devbot/tools.py` that renders an
  indented directory tree starting at `path`, pruning `SKIP_DIRS` and (reusing
  the Phase-1-era gitignore helpers) gitignored entries, limited to `max_depth`.
- Cap total entries (e.g. 500) with a truncation note, like `find_files`.
- Register it in `TOOL_SCHEMAS`/`_TOOL_HANDLERS` (read-only).

**Done when:** `tree` shows a depth-limited structure that excludes skipped and
gitignored paths; new tests verify pruning, depth limiting, and the entry cap.
`pytest -q` green.

---

## Phase 3 — Diff preview on edits

`write_file`/`edit_file` report only a char count. Show what actually changed.

**What to do:**
- In `devbot/tools.py`, after a successful `edit_file` (and `write_file` on an
  existing file), compute a unified diff of old vs new content with
  `difflib.unified_diff`, clip it to a sane size, and append it to the returned
  message (e.g. a fenced ```diff block).
- For brand-new files, just report the line count (no diff).
- Keep the existing success text so nothing that parses it breaks.

**Done when:** editing a file returns a unified diff of the change; creating a
new file does not error; the diff is clipped for huge changes. New tests cover
edit-diff, new-file (no diff), and clipping. `pytest -q` green.

---

## Phase 4 — Edit checkpoints and undo

Autonomous edits have no safety net. Add lightweight per-edit backups + undo.

**What to do:**
- Before `write_file`/`edit_file` modifies an existing file, copy the current
  contents to `.devbot/backups/<timestamp>/<relpath>` (create dirs as needed).
- Add an `undo_last_edit(root)` tool that restores the most recent backup set
  (the newest timestamped folder) and reports what it restored.
- Add a `/undo` REPL command in `devbot/cli.py` that calls it.
- Keep only the most recent N backup sets (e.g. 20) to avoid unbounded growth;
  `.devbot/` is already gitignored.

**Done when:** an edit creates a backup, `undo_last_edit` restores the prior
content, and old backups are pruned past the limit. New tests cover
backup-on-edit, restore, and pruning (no network). `pytest -q` green.

---

## Phase 5 — Project config file

Configuration is env-var only. Add an optional project config file.

**What to do:**
- Load `.devbot/config.toml` (stdlib `tomllib`) at agent startup, if present.
  Recognized keys mirror existing settings: `model`, `max_parallel`,
  `token_budget`, `global_budget`, `loop_limit`, `compress_model`,
  `mega_warn_threshold`, `pipeline_rounds`.
- Precedence: real environment variables win over the config file, which wins
  over built-in defaults. (Do NOT add new env vars — reuse the existing ones.)
- Document the config file and precedence in the README.

**Done when:** a `.devbot/config.toml` value is applied when the matching env
var is unset, and ignored when the env var is set; missing/malformed config is
handled gracefully. New tests cover precedence and the malformed case.
`pytest -q` green.

---

## Cross-cutting rules
- Every phase adds tests and leaves `pytest -q` green.
- Use `pipeline` for all code changes; mock agents in tests (no live API).
- Keep README↔code env-var parity (test_phase5).
- Preserve the path sandbox, the shell allow/deny model, and the approval rule.
