# DevBot improvement plan: parallel UX + mass parallelism

Three independent, shippable phases. Build in this order. After each phase,
run a syntax/compile check and (where possible) a smoke test before moving on.
Prefer `deepseek-v4-flash` while developing to keep cost down.

---

## Phase 1 — Suppress thinking tokens (smallest, do first)

**Goal:** stop streaming the model's full chain-of-thought to the terminal by
default; show a compact, collapsing indicator instead.

**Where:**
- `devbot/agent.py` — `Agent.__init__` and `Agent._stream_once` (the
  `reasoning_content` branch).
- `devbot/cli.py` — a `/think` toggle and a line in `COMMANDS_HELP`.

**What to do:**
- Add `self.show_reasoning` (default `False`), initialized from env var
  `DEVBOT_SHOW_REASONING` (truthy = on).
- In `_stream_once`, when `reasoning_content` arrives:
  - If `show_reasoning` is **off**: do NOT stream the CoT text. Instead update a
    single in-place line like `💭 thinking… (N tokens)` and clear it when the
    first real `content` delta arrives.
  - If **on**: keep the current behavior (full dimmed CoT stream).
- Reasoning must continue to be excluded from `text_parts` (the saved answer) —
  this is display-only.
- Add a `/think` REPL command that toggles `agent.show_reasoning` and prints the
  new state; add it to `COMMANDS_HELP`.

**Done when:** a run shows the `thinking…` indicator (not the full CoT) by
default; `/think` flips it; the model's saved answer is unchanged.

---

## Phase 2 — Mass-parallelism plumbing (config + safety, no UI)

**Goal:** let megaswarm fan out to N agents safely, without tripping rate limits
or exhausting the HTTP connection pool.

**Where:**
- `devbot/swarm.py` — `run_megaswarm`, `megadelegate_schema`.
- `devbot/agent.py` — the shared `OpenAI` client construction.

**What to do:**
- Generalize `megadelegate` to accept `n` (number of parallel agents, default 3)
  and an optional `role` (default the existing trio behavior). Launch N
  sub-agents; the reviewer still synthesizes all outputs.
- Cap concurrency: `max_workers = min(n, DEVBOT_MAX_PARALLEL)` where
  `DEVBOT_MAX_PARALLEL` defaults to 8. Add a `threading.Semaphore` that caps
  in-flight API calls so we respect DeepSeek's concurrency limits
  (deepseek-v4-pro = 500, deepseek-v4-flash = 2500).
- Tune the shared client for many connections:
  `OpenAI(http_client=httpx.Client(limits=httpx.Limits(max_connections=64,
  max_keepalive_connections=32)), timeout=httpx.Timeout(120.0, connect=10.0))`.
- Add a `DEVBOT_TOKEN_BUDGET` guard: once the manager's cumulative tokens exceed
  it, stop spawning new sub-agents and report the partial result.
- Keep per-agent failure isolation (one agent's exception must not kill the run).

**Done when:** `megadelegate(task, n=10)` runs without 429 errors or
connection-pool warnings, and the token budget aborts cleanly when exceeded.

---

## Phase 3 — Live parallel dashboard (biggest, do last)

**Goal:** show a readable, live, in-place view of what every parallel agent is
doing — instead of either silence or garbled interleaved output.

**Where:**
- `devbot/swarm.py` — a new `ParallelMonitor` class; re-point sub-agent hooks in
  `_run_one` to write to it instead of printing.

**What to do:**
- `ParallelMonitor` holds a thread-safe `{label: AgentStatus}` where
  `AgentStatus = {phase, current_tool, last_snippet, tokens, elapsed, state}`.
- Re-point each sub-agent's hooks:
  - `on_tool_start` → set `current_tool` + a short snippet (e.g.
    `⏺ edit_file foo.py`).
  - `on_text` → keep only the last ~60 chars as a rolling snippet (do NOT
    accumulate full output).
  - `on_tool_end` → update state.
- A renderer redraws a fixed N-line block, throttled to ~10 Hz, using ANSI
  cursor moves: `\x1b[{N}A` (up N lines) then `\x1b[2K` (clear) per row.
- When N is large, collapse to a summary line (e.g. "12 agents: 8 running, 3
  done, 1 failed") plus the few most-recently-active agents.

**Must handle:**
- Pause the renderer while `_CONFIRM_LOCK` holds an interactive prompt (a prompt
  drawn mid-redraw corrupts the screen).
- Truncate each line to the terminal width.
- Non-TTY fallback (CI / piped output): print plain `[label] finished` lines
  instead of cursor manipulation.

**Done when:** a live 3-agent megaswarm shows one updating line per agent with no
garble, and Ctrl-C / approval prompts don't corrupt the display.

---

## Cross-cutting notes

- Phases 1 and 3 both need an in-place "transient line" rendering helper (cursor
  move + clear). Build that helper once in Phase 1 and reuse it in Phase 3.
- Phase 3 is the only phase that genuinely needs a live multi-agent run to
  verify; Phases 1 and 2 are unit-checkable. Test the dashboard with a cheap
  throwaway task on `deepseek-v4-flash`, not `-pro`.
- Respect the existing approval-safety rule: parallel sub-agents auto-approve
  dangerous tools only when the manager was started with `-y`; otherwise they
  decline rather than block on stdin.
