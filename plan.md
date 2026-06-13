# DevBot improvement plan (autopilot-ready)

Run with `devbot --run-plan plan.md`. Each `## Phase` is independent and
self-contained. Every phase MUST add or update tests under `tests/` and leave
`pytest -q` green (CI runs it). Use the `pipeline` tool for all code changes;
never bare write/edit as the manager. No live API calls in tests — mock agents
and use `tmp_path`/`monkeypatch`. Preserve the path sandbox and approval-safety
invariant.

These are genuinely new items — testing/CI, shell sandbox, sessions, cost
controls, megaswarm pipeline/divide, dashboard, reasoning suppression,
recursive glob, JSONL logging, readline UX, and the global token budget already
exist. Do not redo them.

---

## Phase 1 — .gitignore-aware search

`grep` and `find_files` in `devbot/tools.py` skip `SKIP_DIRS` but still return
files the user has gitignored (build output, secrets, caches).

**What to do:**
- Add a helper that reads the project's `.gitignore` (root-level is enough) and
  builds a matcher for its patterns (reuse `_glob_match` / `fnmatch`; support
  `dir/`, `*.ext`, and bare names; ignore comments and blank lines).
- In `grep` and `find_files`, skip any path whose relative form matches a
  gitignore pattern, in addition to the existing `SKIP_DIRS` pruning.
- Add an opt-out: a `respect_gitignore` argument (default True) on both tools,
  surfaced in their schemas.
- Keep it dependency-free and fast (parse `.gitignore` once per call).

**Done when:** a gitignored file is excluded from `grep`/`find_files` by default
and included when `respect_gitignore=False`; non-ignored files are unaffected.
New tests cover both. `pytest -q` green.

---

## Phase 2 — Stuck-loop detection

In long autonomous runs an agent can repeat the same failing tool call and burn
tokens. `Agent.run()` in `devbot/agent.py` has no guard against this.

**What to do:**
- Track a short history of recent tool calls (name + a hash of the JSON args).
- If the SAME tool call (name+args) occurs `DEVBOT_LOOP_LIMIT` times in a row
  (default 3), stop the loop with a clear message and return, instead of
  continuing to `max_turns`.
- Also stop if the same tool returns the identical error string that many times
  in a row.
- Make the limit configurable via the `DEVBOT_LOOP_LIMIT` env var (0 disables).

**Done when:** a simulated repeated identical tool call halts after the limit
with a clear message; normal varied tool use is unaffected. New tests drive the
loop with a stubbed `_stream_once` (no network). `pytest -q` green.

---

## Phase 3 — REPL quality-of-life: tab completion, /tools, /cost

The interactive REPL has no completion and no quick way to see tools or cost.

**What to do:**
- When readline is available, register a completer for slash-commands (`/help`,
  `/clear`, `/stats`, `/model`, `/think`, `/swarm`, `/megaswarm`, `/resume`,
  `/sessions`, `/exit`) so Tab completes them.
- Add a `/tools` command that lists the agent's currently available tool names
  (from `agent.tool_schemas`).
- Add a `/cost` command that prints `agent.estimated_cost()` formatted as
  `$X.XX` plus the session token total.
- Update `COMMANDS_HELP` and the README REPL section.

**Done when:** `/tools` and `/cost` work, the completer returns the right
candidates for a given prefix, and tests cover the completer function and the
two new command handlers (no network). `pytest -q` green.

---

## Phase 4 — Session export to Markdown

Sessions persist as JSON but there's no human-readable export.

**What to do:**
- Add `export_markdown(agent, path)` in `devbot/session.py` that renders the
  conversation as Markdown: a header (model, mode, tokens, est. cost), then each
  user/assistant turn, with tool calls shown as fenced blocks and tool results
  truncated to a sane length.
- Add an `/export [path]` REPL command (default
  `.devbot/session-<id>.md`) wired in `devbot/cli.py`.
- Redact anything matching the API-key pattern in the output.

**Done when:** `/export` writes a valid Markdown file capturing the
conversation; secrets are redacted. New tests build a fake agent and assert the
file content (no network). `pytest -q` green.

---

## Phase 5 — Accurate cost via cache-hit accounting

`estimated_cost()` uses a flat 50/50 input/output split and ignores DeepSeek's
much-cheaper cache-hit input tokens, so estimates run high.

**What to do:**
- In `_stream_once`, capture the usage breakdown the API returns
  (`prompt_tokens`, `completion_tokens`, and cache-hit/miss prompt tokens if
  present — e.g. `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`).
  Accumulate these on the agent.
- Add cache-hit input rates to `MODEL_PRICING` and make `estimated_cost()` use
  the real input/output/cache breakdown when available, falling back to the
  current heuristic when it isn't.
- Persist the new counters in `session.py` (defensively, with `getattr`).

**Done when:** with a usage breakdown present, `estimated_cost()` reflects
cache-hit pricing; with only totals, it falls back to the old behaviour. New
tests cover both paths. `pytest -q` green.

---

## Cross-cutting rules
- Every phase adds tests and leaves `pytest -q` green.
- Use `pipeline` for all code changes; mock agents in tests (no live API).
- Document any new env var in the README config table (test_phase5 enforces
  README↔code parity, so keep them in sync).
- Preserve the path sandbox, the shell allow/deny model, and the approval rule.
