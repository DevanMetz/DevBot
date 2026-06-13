# DevBot improvement plan (autopilot-ready)

Each `## Phase` below is independent and self-contained — run them with
`devbot --run-plan plan.md`. After each phase the test suite must pass, so
**every phase MUST add or update tests** under `tests/` and leave `pytest -q`
green. Use the `pipeline` tool for all code changes. Do not start a phase other
than the one assigned. Keep changes small and focused.

These are genuinely pending items — the test suite, CI, shell sandbox, session
persistence, cost controls, megaswarm pipeline/divide-and-conquer, the live
dashboard, and reasoning suppression already exist; do not redo them.

---

## Phase 1 — Recursive glob (`**`) in grep and find_files

`grep` and `find_files` in `devbot/tools.py` match filenames with
`Path(name).match(glob)`, which does NOT support `**` recursive patterns or
path-segment globs like `src/**/*.py`. They already walk recursively, but the
glob only ever sees the bare filename.

**What to do:**
- Make the `glob` argument support both bare-name patterns (`*.py`) and
  path-relative recursive patterns (`**/*.py`, `devbot/*.py`).
- Implement by matching the path **relative to the search base** with `fnmatch`
  (translate `**` appropriately) or `pathlib.PurePath.full_match` if available;
  fall back to filename matching when the pattern has no `/`.
- Keep the existing `SKIP_DIRS` pruning and result caps unchanged.
- Update the tool schema descriptions for `grep`/`find_files` to mention `**`.

**Done when:** `find_files("**/*.py")` and `grep(pattern, glob="devbot/*.py")`
match nested paths correctly, plain `*.py` still works, and new tests in
`tests/` cover both bare and recursive patterns. `pytest -q` green.

---

## Phase 2 — Optional structured logging (`DEVBOT_LOG`)

There is no machine-readable record of what DevBot did — only the pretty
terminal output.

**What to do:**
- Add a small logging helper (e.g. `devbot/devlog.py`) that, when the env var
  `DEVBOT_LOG` is set to a file path, appends one JSON object per line (JSONL)
  for: each tool call (name, args summary, result length, ok/error) and each
  assistant turn (model, prompt tokens, total tokens).
- Wire it into the tool-dispatch path in `devbot/agent.py` behind the env var so
  there is zero overhead and no behavior change when unset.
- Never log secrets: redact any value that looks like an API key.

**Done when:** with `DEVBOT_LOG` set, a valid JSONL file is written with one
record per tool call; with it unset, nothing is written and behavior is
unchanged. New tests cover the enabled and disabled paths (no network).

---

## Phase 3 — readline UX on Windows

Every startup prints `[devbot] readline not available …`, which is noisy.

**What to do:**
- In `devbot/cli.py`, only emit the readline warning ONCE per machine (e.g.
  suppress it after first shown via a marker file under the user config dir), or
  downgrade it to a single concise hint that also says
  `pip install pyreadline3` to enable history.
- Document the optional `pyreadline3` install in the README setup section
  (it is already an optional `[win]` extra in `pyproject.toml`).

**Done when:** a fresh interactive start shows at most a one-line hint (or none
after first run), the REPL still works without readline, and a test verifies the
warning logic (mock the readline-missing path). `pytest -q` green.

---

## Phase 4 — Session-wide token budget for autopilot/megaswarm

`DEVBOT_TOKEN_BUDGET` is enforced per-Agent, but autopilot and megaswarm spawn
fresh agents, so the budget resets and never caps the whole run.

**What to do:**
- Add a process-wide cumulative token counter (module-level in `devbot/agent.py`,
  incremented wherever `total_tokens` is updated) and a `DEVBOT_GLOBAL_BUDGET`
  env var.
- When the global budget is exceeded, new `Agent.run()` calls should stop early
  with a clear message (same pattern as the existing per-session budget).
- Have `devbot/autopilot.py` check the global budget between phases and stop
  cleanly if exceeded, reporting how many phases completed.

**Done when:** setting `DEVBOT_GLOBAL_BUDGET` halts work across multiple agents /
phases; unset means unlimited. New tests cover the counter and the early-stop
(mock the agents — no network).

---

## Phase 5 — Documentation refresh

The README predates swarm/megaswarm, pipeline, sessions, cost controls, and
autopilot.

**What to do:**
- Update `README.md` to document: all env vars (`DEVBOT_MODEL`,
  `DEVBOT_MAX_TURNS`, `DEVBOT_MAX_PARALLEL`, `DEVBOT_TOKEN_BUDGET`,
  `DEVBOT_GLOBAL_BUDGET`, `DEVBOT_COMPRESS_MODEL`, `DEVBOT_MEGA_WARN_THRESHOLD`,
  `DEVBOT_SHOW_REASONING`, `DEVBOT_ALLOW_SHELL`, `DEVBOT_LOG`); the full tool
  list; swarm vs megaswarm vs pipeline; `--run-plan` autopilot; and `--resume`.
- Add a short "Safety model" section (sandboxed paths, shell allow/deny list,
  approval gates, mandatory review pipeline).
- Keep it accurate — verify each documented flag actually exists in the code.

**Done when:** the README reflects the current feature set with no invented
flags. (This phase changes docs/tests only; ensure `pytest -q` still passes.)

---

## Cross-cutting rules
- Every phase adds tests and leaves `pytest -q` green (CI runs them).
- Use `pipeline` for all code changes; never bare write/edit as the manager.
- No live API calls in tests — mock agents and use `tmp_path`.
- Preserve the approval-safety invariant and the path sandbox.
