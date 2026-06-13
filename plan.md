# DevBot self-improvement plan

Phases are independent and ordered by value. After each phase: run
`python -m pytest -q` (once Phase 1 lands) plus a syntax/compile check, and
verify before moving on. Prefer `deepseek-v4-flash` while developing to keep
cost down. Commit after each phase.

---

## Phase 1 — Make the project testable (do first)

The project has `tests/` but no test runner config, and the core file-mutating
logic is almost entirely untested. Lock it down before changing anything else.

**What to do:**
- Add `pytest` to `pyproject.toml` as an optional `[dev]` dependency and a
  `[tool.pytest.ini_options]` section (`testpaths = ["tests"]`).
- Write unit tests (no network) for the high-risk pure logic in `tools.py`:
  - `_resolve`: relative path inside root OK; `..` escape and absolute paths
    outside root raise `PathError`; symlink escape blocked.
  - `edit_file`: empty `old_string`, missing string (with closest-line hint),
    non-unique without `replace_all`, successful single + `replace_all`.
  - `read_file`: negative offset clamps, offset past EOF errors, truncation
    header appears only when content is clipped.
  - `grep` / `find_files`: `SKIP_DIRS` pruning; match caps.
  - `run_command`: `BLOCKED_PATTERNS` rejects dangerous commands.
  - `dispatch`: unknown tool, null-coerced args.
- Add tests for `agent._compress_conversation` keep-index logic (no orphaned
  tool messages) and `_load_dotenv`.
- Add a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs
  `pytest -q` on push/PR (Python 3.10–3.13).

**Done when:** `pytest -q` passes locally and the CI workflow is committed.

---

## Phase 2 — Harden the shell sandbox

`run_command` uses `shell=True` gated only by a regex blocklist — a deny-list is
the wrong default. Move toward an allow-list / explicit-confirmation model.

**What to do:**
- Add a `DEVBOT_ALLOW_SHELL` posture: default to requiring per-command approval
  even in `-y` mode for commands not on a safe allow-list (git status, ls,
  pytest, python, npm test, etc.).
- Surface the full command in the approval prompt (already partly there) and
  show *why* it was flagged (matched blocklist vs. not on allow-list).
- Fix the Windows encoding bug: `run_command` decodes subprocess output as UTF-8
  but `cmd.exe` emits cp1252 — detect the console encoding or use
  `errors="replace"` consistently and document the limitation.
- Add tests for the allow-list / block-list decision logic.

**Done when:** a non-allow-listed command prompts even under `-y`; allow-listed
ones run freely; tests cover the decision matrix.

---

## Phase 3 — Conversation persistence & resume

Long sessions (and expensive megaswarm runs) are lost on exit.

**What to do:**
- Persist the conversation (`messages`, model, mode, token totals) to
  `.devbot/session-<id>.json` on each turn.
- Add `--resume [id]` and a `/resume` command to reload the latest/selected
  session.
- Add `/sessions` to list saved sessions with timestamps and token counts.
- Respect the sandbox: store under the project root; add `.devbot/` to
  `.gitignore`.

**Done when:** quitting mid-task and `devbot --resume` restores the exact
conversation and stats.

---

## Phase 4 — Cost & context controls

Megaswarm runs have hit 1.5M tokens; users need visibility and brakes.

**What to do:**
- Compression: use a cheaper model for summarization (configurable,
  default `deepseek-v4-flash`) instead of the active model.
- Add a per-session `DEVBOT_TOKEN_BUDGET` enforced for the *whole* session (not
  just one megaswarm), with a clear stop + summary when hit.
- Show a running cost estimate in `/stats` using the model's per-1M pricing.
- Warn before a `megadelegate` that would launch more than N agents.

**Done when:** `/stats` shows estimated $ cost; the session budget halts work
with a clear message; compression uses the cheap model.

---

## Phase 5 — Polish & papercuts

- Windows readline: either bundle `pyreadline3` install guidance prominently or
  implement a minimal history fallback so the startup warning isn't the first
  thing users see.
- `_resolve`/glob: `Path.match` doesn't support `**`; document or implement
  recursive glob in `grep`/`find_files`.
- Structured logging: optional `DEVBOT_LOG=path` to tee tool calls + errors to a
  file (JSONL), separate from the pretty terminal output.
- Remove the redundant megaswarm semaphore (the `ThreadPoolExecutor` already
  caps concurrency) or document why it stays.

**Done when:** each papercut is fixed or explicitly documented as a known
limitation in the README.

---

## Cross-cutting rules
- Keep the approval-safety invariant: parallel sub-agents auto-approve dangerous
  tools only with `-y`; otherwise they decline rather than block on stdin.
- Every new feature ships with tests (post Phase 1) and a README update.
- Verify with a cheap `deepseek-v4-flash` run before any `deepseek-v4-pro` run.
