# DevBot auto-evolve mode (autopilot-ready plan)

Goal: a **bounded** self-evolving loop — DevBot plans its own phases, critiques
them, and executes them on a dedicated git branch with per-phase commits, hard
stops, and CI as an independent gate. **Never touches main; never auto-merges.**

Build it with the existing autopilot: `devbot --run-plan plan.md`. Each `## Phase`
is independent and self-contained. Every phase MUST add/update tests under
`tests/` and leave `pytest -q` green. Use the `pipeline` tool for all code
changes. **No live API calls and no real network/git side effects in tests** —
mock agent runners, and for git use a throwaway repo created under `tmp_path`.
Any new `DEVBOT_*` env var MUST be added to the README config table
(`test_phase5` enforces README↔code parity).

Design invariants every phase must preserve:
- The loop runs on a branch like `autopilot/<YYYY-MM-DD-HHMM>`, never `main`/`master`.
- Each accepted phase = one commit on that branch (the real recovery unit).
- Hard stops: global token budget (exists), max-phases-per-run, wall-clock
  limit, and stop-on-red (exists). Any one trips → halt cleanly.
- No `git merge`/`git push` to main, ever. Pushing the work branch is opt-in.

---

## Phase 1 — Git checkpoint helpers

A new module `devbot/gitcheckpoint.py` with small, well-tested git wrappers
(via `subprocess`, no extra deps).

**What to do:**
- `current_branch(root) -> str` — the checked-out branch name.
- `is_clean(root) -> bool` — working tree has no uncommitted changes.
- `ensure_branch(root, name) -> None` — create+checkout `name` if it doesn't
  exist, else checkout it. Refuse (raise) if `name` is `main`/`master`.
- `commit_all(root, message) -> str | None` — `git add -A` + commit; return the
  short SHA, or None if there was nothing to commit.
- All functions return/raise predictably; never operate outside `root`.

**Done when:** in a `tmp_path` git repo, the helpers create a branch, commit a
change (returning a SHA), report branch/cleanliness correctly, and `ensure_branch`
refuses `main`. New tests cover each. `pytest -q` green.

---

## Phase 2 — Hard-stop controller

A `devbot/evolve_limits.py` module that centralizes all stop conditions.

**What to do:**
- A `StopController` (or functions) tracking: a wall-clock deadline, a
  max-phases counter, and hooks into the existing global-token-budget check
  (`agent.check_global_budget_exceeded`) and stop-on-red.
- Config via env vars: `DEVBOT_EVOLVE_MAX_PHASES` (default 20),
  `DEVBOT_EVOLVE_TIME_LIMIT` (minutes, default 0 = unlimited). Reuse
  `DEVBOT_GLOBAL_BUDGET` for tokens (do not add a new token var).
- `should_stop() -> tuple[bool, str]` — returns (True, human-readable reason)
  when any limit is hit; `record_phase()` increments the phase counter.

**Done when:** the controller reports the correct stop reason for each trigger
(max phases, time limit, global budget) and keeps going otherwise. New tests
drive each path with monkeypatched time/counters (no real waiting). Document the
two new env vars in the README. `pytest -q` green.

---

## Phase 3 — Planner + critic

The brains: generate the next 5 phases, then filter them for real value.

**What to do:**
- In a new `devbot/evolve_planner.py`:
  - `generate_plan(manager, context) -> list[dict]` — uses a planner agent to
    propose up to 5 phases (title + body), returned via the existing
    `autopilot.parse_phases` format. `context` includes the repo's current state
    (e.g. an `outline`/`tree` summary and the README) so it proposes relevant work.
  - `critique_plan(manager, phases) -> list[dict]` — a critic agent scores each
    proposed phase and **drops** low-value ones: pure refactors, speculative
    features, anything not clearly improving correctness/usefulness/safety.
    Returns the surviving phases (possibly fewer than 5), each with a one-line
    justification.
- Both must be robust to malformed model output (return [] rather than crash).

**Done when:** with mocked agent runners, `generate_plan` parses proposed phases
and `critique_plan` removes the ones the critic rejects (and keeps the rest).
New tests cover parsing, rejection, and the malformed-output fallback. `pytest
-q` green.

---

## Phase 4 — The auto-evolve driver + `--auto-evolve`

Tie it together into the actual loop, reusing existing autopilot per-phase
execution.

**What to do:**
- In `devbot/autopilot.py` (or a new `evolve.py`): `run_evolve(root, model)`:
  1. Refuse to start unless the working tree is clean and the current branch is
     not main (else create/checkout `autopilot/<timestamp>` via Phase 1 helpers).
  2. Loop: `generate_plan` → `critique_plan` → for each surviving phase, run the
     existing implement→verify cycle (pipeline, one fix round, stop-on-red). On
     green, `commit_all` to the branch (per-phase commit). Call
     `record_phase()`; check `should_stop()` before each phase and between plans.
  3. When a plan's phases are done and no stop hit, generate the NEXT plan and
     continue. Halt cleanly on any stop condition, printing a summary.
- Add `--auto-evolve` to the CLI (implies unattended/auto-approve), with the
  loud unattended warning.

**Done when:** with planner/critic/agent/verify/git all mocked, `run_evolve`
commits one checkpoint per green phase on a non-main branch, stops on the first
hard limit, and never commits to main. New tests assert: per-phase commits,
stop-on-red, stop-on-cap, branch isolation. `pytest -q` green.

---

## Phase 5 — Safety rails, opt-in push, and summary report

Harden the loop and make the result reviewable.

**What to do:**
- Guard rails: hard-refuse if asked to run on `main`/`master`; refuse if the
  working tree is dirty at start; never run `git merge`/`git push` to main.
- Opt-in CI gate: `--evolve-push` flag (or `DEVBOT_EVOLVE_PUSH=1`) that, after
  each phase commit, pushes the work branch so CI runs as an independent oracle.
  Off by default (pushing is outward-facing).
- End-of-run summary: phases completed, branch name + per-phase commit SHAs,
  total tokens + estimated cost, and the stop reason. Print it and also write it
  to `.devbot/evolve-<timestamp>.md`.
- README: document `--auto-evolve`, `--evolve-push`, the new env vars, the
  branch/no-auto-merge model, and how to review/cherry-pick/discard the branch.

**Done when:** the safety guards reject main/dirty-tree starts (tested), the
summary report is generated with the expected fields (tested with mocks), and
push stays opt-in. `pytest -q` green; README parity test passes.

---

## Cross-cutting rules
- Every phase adds tests and leaves `pytest -q` green; mock all agents/git/network.
- Use `pipeline` for all code changes; never bare write/edit as the manager.
- Keep README↔code env-var parity (`test_phase5`).
- Never auto-merge to main; pushing the work branch is opt-in only.
- Preserve the path sandbox, shell allow/deny model, and approval invariant.
