"""Agent swarm: a manager agent that delegates tasks to specialist sub-agents.

Pattern: the manager gets a `delegate(role, task)` tool. Each call spawns a
fresh Agent with a specialist system prompt and a restricted tool set, runs its
own agentic loop, and returns the specialist's final answer to the manager.

Specialists never get the delegate tool themselves, so there is no recursion.

Megaswarm (`megadelegate`): delegates the same task to N specialists *in parallel*
(default 3, configurable via the `n` parameter), then hands their outputs to a
reviewer who synthesises a single combined answer for the manager. This gives a
more robust result by triangulating independent analyses.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import sys
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a circular import at runtime (agent.py imports nothing here)
    from .agent import Agent

# All read-only tools, used by review/research style specialists.
_READ_ONLY = ["read_file", "list_dir", "grep", "find_files", "web_search"]

# role -> config. `tools=None` means "all tools"; `model=None` inherits manager's.
SPECIALISTS: dict[str, dict] = {
    "coder": {
        "description": "Writes, edits, and refactors code. Full tool access.",
        "tools": None,
        "model": None,
        "prompt": (
            "You are a specialist CODER agent in a swarm. You receive a single, "
            "self-contained coding task from a manager agent. Inspect the relevant "
            "files, make the change, and verify it (run tests or the script) when "
            "possible. Return a concise summary of exactly what you changed and any "
            "follow-ups the manager should know about."
        ),
    },
    "reviewer": {
        "description": "Reviews code for bugs, edge cases, and style. Read-only.",
        "tools": _READ_ONLY,
        "model": None,
        "prompt": (
            "You are a specialist code REVIEWER in a swarm. You have read-only tools. "
            "Review the code you are asked about for correctness, edge cases, security, "
            "and readability. Be specific: cite file:line and give actionable fixes. "
            "Do not attempt to edit files."
        ),
    },
    "tester": {
        "description": "Runs tests, diagnoses failures, suggests fixes.",
        "tools": _READ_ONLY + ["run_command", "verify"],
        "model": None,
        "prompt": (
            "You are a specialist TESTER in a swarm. Run the project's tests, read the "
            "output, identify the root cause of any failures, and report specific fixes. "
            "You may run commands but should not edit files — leave fixes to the coder."
        ),
    },
    "researcher": {
        "description": "Explores the codebase and answers questions. Read-only.",
        "tools": _READ_ONLY,
        "model": None,
        "prompt": (
            "You are a specialist RESEARCHER in a swarm. Explore the codebase to answer "
            "the manager's question thoroughly. Read relevant files, search for patterns, "
            "and cite specific file:line locations in your answer."
        ),
    },
}


def delegate_schema() -> dict:
    """The `delegate` tool definition added to the manager's tool set."""
    roles = ", ".join(f"{r} ({c['description']})" for r, c in SPECIALISTS.items())
    return {
        "type": "function",
        "function": {
            "name": "delegate",
            "description": (
                "Delegate a self-contained subtask to a specialist sub-agent, which "
                "works independently and returns its final result. Available roles: "
                + roles
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": list(SPECIALISTS),
                        "description": "Which specialist to delegate to.",
                    },
                    "task": {
                        "type": "string",
                        "description": "A clear, self-contained description of the subtask.",
                    },
                },
                "required": ["role", "task"],
            },
        },
    }


def run_specialist(manager: "Agent", role: str, task: str) -> str:
    """Spawn a specialist sub-agent, run the task, return its final answer."""
    from .agent import Agent, TOOL_SCHEMAS  # local import avoids circular dependency

    spec = SPECIALISTS.get(role)
    if spec is None:
        return f"Error: unknown specialist '{role}'. Choose one of: {', '.join(SPECIALISTS)}"

    allowed = spec["tools"]
    schemas = (TOOL_SCHEMAS if allowed is None
               else [s for s in TOOL_SCHEMAS if s["function"]["name"] in allowed])

    sub = Agent(
        root=manager.root,
        model=spec["model"] or manager.model,
        auto_approve=manager.auto_approve,
        system_prompt=spec["prompt"],
        tool_schemas=schemas,
        label=role,
    )
    # Share the already-authenticated client so we don't re-read env / re-auth.
    sub.client = manager.client

    print(f"\n\x1b[35m╭─ delegating to [{role}]\x1b[0m \x1b[90m{task[:100]}\x1b[0m")
    start = time.time()
    answer = sub.run(task)
    elapsed = time.time() - start
    print(f"\n\x1b[35m╰─ [{role}] done in {elapsed:.1f}s\x1b[0m")

    # Roll the sub-agent's token usage up into the manager's session totals.
    manager.total_tokens += sub.total_tokens
    manager.delegation_count += 1
    return _clip_agent_result(answer or "(specialist returned no text)", role)


def run_specialist_with_prompt(manager: "Agent", role: str, system_prompt: str,
                               task: str, label: str | None = None) -> str:
    """Like run_specialist but with a custom system prompt (keeps the role's tools)."""
    from .agent import Agent, TOOL_SCHEMAS  # local import avoids circular dependency

    spec = SPECIALISTS.get(role)
    if spec is None:
        return f"Error: unknown specialist '{role}'."
    allowed = spec["tools"]
    schemas = (TOOL_SCHEMAS if allowed is None
               else [s for s in TOOL_SCHEMAS if s["function"]["name"] in allowed])
    lbl = label or role
    sub = Agent(
        root=manager.root,
        model=spec["model"] or manager.model,
        auto_approve=manager.auto_approve,
        system_prompt=system_prompt,
        tool_schemas=schemas,
        label=lbl,
    )
    sub.client = manager.client

    print(f"\n\x1b[35m╭─ [{lbl}]\x1b[0m \x1b[90m{task[:100]}\x1b[0m")
    start = time.time()
    answer = sub.run(task)
    print(f"\n\x1b[35m╰─ [{lbl}] done in {time.time() - start:.1f}s\x1b[0m")
    manager.total_tokens += sub.total_tokens
    manager.delegation_count += 1
    return _clip_agent_result(answer or "(specialist returned no text)", lbl)


# ---------------------------------------------------------------------------
#  Megaswarm — 3 agents in parallel → reviewer synthesises
# ---------------------------------------------------------------------------

# Which three roles to launch in parallel. The idea is to attack the problem
# from three complementary angles so the reviewer can triangulate.
MEGASWARM_TRIO: tuple[str, str, str] = ("coder", "researcher", "tester")

MEGASWARM_REVIEWER_PROMPT = (
    "You are a synthesis REVIEWER in a megaswarm. Several specialist agents have just "
    "worked on the SAME task independently and produced separate reports. "
    "Your job is to read all outputs carefully and produce a single, coherent, "
    "combined answer for the manager.\n\n"
    "Guidelines:\n"
    "- Identify where the specialists agree — that is the high-confidence core.\n"
    "- Note any disagreements or different approaches and explain the trade-offs.\n"
    "- Merge complementary insights across all reports.\n"
    "- Produce one final, actionable synthesis. Be concise but complete.\n"
    "- If one specialist clearly had the best answer, lean on it but still incorporate "
    "unique insights from the others."
)


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(os.environ.get(name, str(default)) or str(default))
    except ValueError:
        return default
    return max(minimum, value)


def _clip_agent_result(text: str, label: str = "agent result") -> str:
    """Bound sub-agent text before feeding it back into another agent."""
    limit = _env_int("DEVBOT_SPECIALIST_RESULT_LIMIT", 8000, minimum=500)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{label} truncated, {len(text)} chars total]"


# ---------------------------------------------------------------------------
#  ParallelMonitor — live dashboard for megaswarm parallel phase
# ---------------------------------------------------------------------------

class AgentStatus:
    """Live status of one parallel agent."""
    __slots__ = ('label', 'phase', 'current_tool', 'last_snippet',
                 'tokens', 'elapsed', 'state', 'last_update')

    def __init__(self, label: str):
        self.label = label
        self.phase = 'running'       # running | done | failed
        self.current_tool: str | None = None
        self.last_snippet = ''
        self.tokens = 0
        self.elapsed = 0.0
        self.state = 'starting'
        self.last_update = time.time()


class ParallelMonitor:
    """Live dashboard that redraws one status line per parallel agent in-place.

    Uses ANSI cursor moves to update a fixed block of terminal lines at ~10 Hz.
    When there are many agents or stdout is not a TTY, falls back to plain lines.
    """

    # Collapse to summary + top-N when more agents than this.
    _COLLAPSE_THRESHOLD = 8
    _MAX_COLLAPSED_LINES = 5  # summary + up to 4 most-recent agents
    _MIN_TERM_WIDTH = 40      # below this, truncation would slice ANSI escapes

    def __init__(self, labels: list[str]):
        self._labels = labels
        self._n = len(labels)
        self.statuses: dict[str, AgentStatus] = {}
        self._lock = threading.Lock()
        self._render_thread: threading.Thread | None = None
        self._running = False
        self._start_time = 0.0   # set in start()
        self._is_tty = sys.stdout.isatty()
        raw_w = shutil.get_terminal_size().columns if self._is_tty else 120
        self._term_width = max(raw_w, self._MIN_TERM_WIDTH)
        self._collapse = self._n > self._COLLAPSE_THRESHOLD
        self._lines_printed = 0
        self._first_render = True
        # Pick icons the terminal can actually render.
        self._icons = self._choose_icons()

    # ---- public API -----------------------------------------------------------

    def start(self):
        """Print initial blank lines and launch the render thread (TTY only)."""
        if not self._is_tty:
            return
        self._start_time = time.time()
        self._running = True
        lines = self._dashboard_line_count()
        for _ in range(lines):
            print()
        self._lines_printed = lines
        self._render_thread = threading.Thread(target=self._render_loop,
                                               daemon=True)
        self._render_thread.start()

    def stop(self):
        """Stop the render thread, do a final redraw, and move past the block."""
        self._running = False
        if self._render_thread:
            self._render_thread.join(timeout=0.8)
        if self._is_tty:
            self._render()
            # Move cursor below the dashboard block so subsequent prints
            # don't overwrite it.
            sys.stdout.write("\n")
            sys.stdout.flush()

    def update(self, label: str, **kwargs):
        """Thread-safe update of one agent's status fields."""
        now = time.time()
        with self._lock:
            st = self.statuses.get(label)
            if st is None:
                st = self.statuses[label] = AgentStatus(label)
            for k, v in kwargs.items():
                if hasattr(st, k):
                    setattr(st, k, v)
            st.last_update = now
            if 'elapsed' not in kwargs:
                st.elapsed = now - self._start_time

    def get_hooks(self, label: str):
        """Return (on_text, on_tool_start, on_tool_end) wired to this monitor.

        The closures capture *label* so each sub-agent writes to its own slot.
        """

        def on_text(chunk: str):
            with self._lock:
                st = self.statuses.get(label)
                if st is not None:
                    st.last_snippet = (st.last_snippet + chunk)[-60:]

        def on_tool_start(name: str, args: dict):
            preview = json.dumps(args)
            if len(preview) > 60:
                preview = preview[:60] + "..."
            tool_icon = self._icons.get('tool', '~')
            self.update(label, current_tool=f"{tool_icon} {name} {preview}",
                        state='in-tool')

        def on_tool_end(result: str):
            self.update(label, current_tool=None, state='idle')

        return on_text, on_tool_start, on_tool_end

    # ---- internal render logic ------------------------------------------------

    @staticmethod
    def _can_encode(ch: str) -> bool:
        """Return True if *ch* can be encoded by stdout."""
        try:
            ch.encode(sys.stdout.encoding or 'utf-8')
            return True
        except (UnicodeEncodeError, UnicodeError, LookupError):
            return False

    @classmethod
    def _choose_icons(cls) -> dict:
        """Return a dict of phase/state → safe icon character."""
        if cls._can_encode('\u23f3'):       # ⏳
            return {'running': '\u23f3', 'done': '\u2705',
                    'failed': '\u274c', 'tool': '\u23ba'}
        else:
            # ASCII-safe fallbacks for narrow encodings (cp1252, etc.)
            return {'running': '>', 'done': '+', 'failed': '!', 'tool': '~'}

    def _dashboard_line_count(self) -> int:
        """How many terminal lines the dashboard occupies."""
        if self._collapse:
            return self._MAX_COLLAPSED_LINES
        return self._n

    def _render_loop(self):
        """Background thread: redraw at ~10 Hz while running."""
        from .agent import _CONFIRM_LOCK  # hoisted; safe (both modules loaded)
        while self._running:
            # Pause while an interactive approval prompt holds the console
            # lock; drawing mid-prompt corrupts the screen.
            if not _CONFIRM_LOCK.locked():
                try:
                    self._render()
                except Exception:
                    # Don't let a render glitch kill the dashboard thread.
                    pass
            time.sleep(0.1)

    def _render(self):
        """Redraw the entire dashboard block in-place."""
        if not self._is_tty:
            return

        with self._lock:
            lines = self._build_lines()

        # On the very first render the cursor is already at the right spot
        # (just after the blank lines printed by .start()).  Subsequent renders
        # must jump back up to the top of the block.
        if not self._first_render and self._lines_printed:
            sys.stdout.write(f"\x1b[{self._lines_printed}A")
        self._first_render = False

        # Write each status line, clearing to end-of-line first.
        for i, line in enumerate(lines):
            sys.stdout.write(f"\x1b[2K{line}\n")

        # If the block shrank (e.g. collapse threshold crossed), erase the
        # leftover lines from the previous render.
        for _ in range(len(lines), self._lines_printed):
            sys.stdout.write("\x1b[2K\n")

        self._lines_printed = len(lines)
        sys.stdout.flush()

    def _build_lines(self) -> list[str]:
        """Build the dashboard lines from current statuses."""
        if not self.statuses:
            return ["\x1b[90m  Starting...\x1b[0m"]

        if self._collapse:
            return self._build_collapsed()
        return self._build_full()

    def _build_collapsed(self) -> list[str]:
        """Summary line + the few most-recently-active agents."""
        running, done, failed = self._count_phases()
        summary = (
            f"\x1b[35;1m══ {self._n} agents:\x1b[0m "
            f"\x1b[33m{running} running\x1b[0m, "
            f"\x1b[32m{done} done\x1b[0m, "
            f"\x1b[31m{failed} failed\x1b[0m"
        )
        lines = [summary]
        sorted_st = sorted(self.statuses.values(),
                           key=lambda s: (s.last_update, s.label),
                           reverse=True)
        for st in sorted_st[:self._MAX_COLLAPSED_LINES - 1]:
            lines.append(self._format_line(st))
        return lines

    def _build_full(self) -> list[str]:
        """One line per agent (in label order)."""
        lines = []
        for label in self._labels:
            st = self.statuses.get(label)
            if st is not None:
                lines.append(self._format_line(st))
            else:
                lines.append(f" \x1b[90m{label}\x1b[0m  \x1b[90m…\x1b[0m")
        return lines

    def _count_phases(self):
        """Return (running, done, failed) counts."""
        running = sum(1 for s in self.statuses.values() if s.phase == 'running')
        done = sum(1 for s in self.statuses.values() if s.phase == 'done')
        failed = sum(1 for s in self.statuses.values() if s.phase == 'failed')
        return running, done, failed

    def _format_line(self, st: AgentStatus) -> str:
        """Format one agent status line, truncated to terminal width."""
        w = self._term_width

        icon = self._icons.get(st.phase, '?')
        label = st.label[:14]

        # Right-hand info: token count + elapsed seconds
        right = f"{st.tokens:,}t {st.elapsed:.0f}s"

        # Middle: tool name or last text snippet
        middle = ""
        if st.current_tool:
            middle = f" {st.current_tool}"
        elif st.last_snippet:
            clean = st.last_snippet.replace('\n', ' ').replace('\r', '')
            middle = f" {clean[-50:]}"

        # Assemble, leaving space for the right-side info
        base = f" {icon} \x1b[1m{label}\x1b[0m"
        available = w - len(base) - len(right) - 2
        if available > 6 and len(middle) > available:
            middle = middle[:available - 1] + "\u2026"
        elif available <= 6:
            middle = ""
        padding = max(1, w - len(base) - len(middle) - len(right))

        return f"{base}{middle}{' ' * padding}{right}"[:w]


def megadelegate_schema() -> dict:
    """The `megadelegate` tool definition added to the manager's tool set."""
    roles = ", ".join(MEGASWARM_TRIO)
    return {
        "type": "function",
        "function": {
            "name": "megadelegate",
            "description": (
                "MEGASWARM: run specialists IN PARALLEL, then a reviewer combines "
                "their work. Two modes:\n"
                "1. DIVIDE-AND-CONQUER (preferred for building): pass `subtasks` = "
                "a list of INDEPENDENT subtasks that touch DIFFERENT files/areas. "
                "Each runs concurrently (role defaults to coder), then the reviewer "
                "integrates them. Use this to actually get more done in parallel.\n"
                f"2. TRIANGULATION: pass only `task` to run N agents ({roles}) on the "
                "SAME task and synthesise their takes — good for analysis/decisions, "
                "not for parallel edits.\n"
                "Do NOT put interdependent edits to the same file in separate "
                "subtasks — they will conflict."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The overall goal (triangulation: the task all "
                        "agents work; divide mode: a short description of the goal).",
                    },
                    "subtasks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "DIVIDE-AND-CONQUER: list of independent subtasks "
                        "run in parallel (different files/areas). Omit for triangulation.",
                    },
                    "n": {
                        "type": "integer",
                        "description": "Triangulation only: number of parallel agents "
                        "(default 3, capped by DEVBOT_MAX_PARALLEL, default 8).",
                    },
                    "role": {
                        "type": "string",
                        "enum": list(SPECIALISTS),
                        "description": "Role for the agents (divide mode: all subtasks; "
                        "triangulation: N copies of this role instead of the trio).",
                    },
                },
                "required": ["task"],
            },
        },
    }


def run_megaswarm(manager: "Agent", task: str, n: int = 3,
                  role: str | None = None,
                  subtasks: list | None = None) -> str:
    """Run specialists in parallel, then hand outputs to a reviewer.

    Two modes:

    * **Divide-and-conquer** (``subtasks`` given): each subtask is a *distinct*
      unit of work run concurrently by its own agent (role = *role* or "coder").
      The reviewer integrates the pieces. Use for independent work touching
      different files — real throughput parallelism.
    * **Triangulation** (no ``subtasks``): launch N agents on the *same* task
      and the reviewer synthesises their independent takes.
    """
    from .agent import Agent, TOOL_SCHEMAS  # local import avoids circular dependency

    divide = subtasks is not None
    if divide:
        task_list = [str(s) for s in subtasks if str(s).strip()]
        if not task_list:
            return "Error: subtasks list was empty."
        chosen = role if role in SPECIALISTS else "coder"
        role_list = [chosen] * len(task_list)
        n = len(task_list)
    else:
        # ---- Guard n ------------------------------------------------------
        n = max(1, int(n))
        # ---- Resolve the list of roles to launch --------------------------
        if role is not None:
            if role not in SPECIALISTS:
                return f"Error: unknown specialist '{role}'. Choose one of: {', '.join(SPECIALISTS)}"
            role_list = [role] * n
        else:
            role_list = [MEGASWARM_TRIO[i % len(MEGASWARM_TRIO)] for i in range(n)]
        task_list = [task] * n

    # ---- Pre-compute unique display labels ----------------------------------
    # So the dashboard can show "coder", "coder-2", "coder-3", … from the start.
    label_counts: dict[str, int] = {}
    labels: list[str] = []
    for r in role_list:
        if r in label_counts:
            label_counts[r] += 1
            labels.append(f"{r}-{label_counts[r]}")
        else:
            label_counts[r] = 1
            labels.append(r)

    # ---- Megadelegate warning -----------------------------------------------
    warn_threshold = int(os.environ.get("DEVBOT_MEGA_WARN_THRESHOLD", "5"))
    if n > warn_threshold:
        # Pick a safe warning character, matching ParallelMonitor._choose_icons.
        warn_icon = "\u26a0"  # ⚠
        try:
            warn_icon.encode(sys.stdout.encoding or "utf-8")
        except (UnicodeEncodeError, UnicodeError, LookupError):
            warn_icon = "!!"
        print(f"\x1b[33m[devbot] {warn_icon} Launching {n} agents — "
              f"this may be expensive. "
              f"Set DEVBOT_MEGA_WARN_THRESHOLD to suppress.\x1b[0m")

    # ---- Concurrency cap & semaphore ---------------------------------------
    max_parallel = int(os.environ.get("DEVBOT_MAX_PARALLEL", "8"))
    max_workers = min(n, max_parallel)
    sem = threading.Semaphore(max_workers)

    # ---- Token budget -------------------------------------------------------
    token_budget = int(os.environ.get("DEVBOT_TOKEN_BUDGET", "0"))

    # ---- Live dashboard (TTY only) ------------------------------------------
    monitor: ParallelMonitor | None = None
    if sys.stdout.isatty():
        monitor = ParallelMonitor(labels)

    mode_desc = "divide" if divide else "triangulate"
    label_desc = f"role={role}," if role else ""
    print(f"\n\x1b[35;1m╔═ MEGASWARM [{mode_desc}, n={n}, {label_desc} "
          f"workers={max_workers}]\x1b[0m \x1b[90m{task[:100]}\x1b[0m")

    if monitor is not None:
        monitor.start()

    # ---- Phase 1: run specialists in parallel ------------------------------
    start = time.time()

    def _run_one(_role: str, _sem: threading.Semaphore,
                 _label: str, _task: str) -> tuple[str, str, int]:
        """Run a single specialist on its own task; return (role, answer, tokens)."""
        spec = SPECIALISTS[_role]
        allowed = spec["tools"]
        schemas = (TOOL_SCHEMAS if allowed is None
                   else [s for s in TOOL_SCHEMAS if s["function"]["name"] in allowed])
        sub = Agent(
            root=manager.root,
            model=spec["model"] or manager.model,
            auto_approve=manager.auto_approve,
            system_prompt=spec["prompt"],
            tool_schemas=schemas,
            label=_label,
        )
        sub.client = manager.client

        # Wire hooks to the live dashboard when available; otherwise silence
        # them to avoid garbled interleaved output in non-TTY mode.
        if monitor is not None:
            sub.on_text, sub.on_tool_start, sub.on_tool_end = monitor.get_hooks(_label)
            monitor.update(_label, phase='running')
        else:
            sub.on_text = lambda chunk: None
            sub.on_tool_start = lambda name, args: None
            sub.on_tool_end = lambda result: None
        sub._transient_line = lambda text, clear=False: None  # silence thinking indicator

        # Interactive approval can't work in the parallel phase (agents would
        # race on stdin). Honor the manager's setting instead: if the user ran
        # with -y, auto-approve; otherwise DECLINE dangerous tools rather than
        # silently running file writes / shell commands without consent.
        if manager.auto_approve:
            sub.confirm = lambda name, args: True
        else:
            sub.confirm = lambda name, args: False
        try:
            _sem.acquire()
            answer = sub.run(_task)
            if monitor is not None:
                monitor.update(_label, phase='done', tokens=sub.total_tokens)
        except Exception as exc:
            answer = f"ERROR: {type(exc).__name__}: {exc}"
            if monitor is not None:
                monitor.update(_label, phase='failed')
        finally:
            _sem.release()
        return (
            _role,
            _clip_agent_result(answer or "(specialist returned no text)", _label),
            sub.total_tokens,
        )

    results: dict[str, str] = {}
    skipped = 0
    budget_msg = ""  # deferred budget-exhausted message (printed after dashboard)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures: dict[concurrent.futures.Future, str] = {}
        for r, lbl, tk in zip(role_list, labels, task_list):
            futures[pool.submit(_run_one, r, sem, lbl, tk)] = lbl

        for fut in concurrent.futures.as_completed(futures):
            try:
                _, answer, tokens = fut.result()
            except concurrent.futures.CancelledError:
                continue  # future was cancelled by token-budget guard
            key = futures[fut]  # pre-computed unique label
            results[key] = answer
            manager.total_tokens += tokens
            manager.delegation_count += 1

            if monitor is None:
                # Non-TTY fallback: print a plain finished line.
                print(f"\x1b[35m  ╟─ [{key}] finished ({tokens:,} tokens)\x1b[0m")

            # Check token budget after each agent completes.
            # If exceeded, cancel remaining unfinished futures so we don't
            # burn more tokens on agents that haven't started yet.
            if token_budget > 0 and manager.total_tokens >= token_budget:
                for f in futures:
                    if not f.done():
                        f.cancel()
                        # Mark skipped agents in the monitor
                        if monitor is not None:
                            monitor.update(futures[f], phase='failed',
                                           state='skipped (budget)')
                        skipped += 1
                if skipped:
                    budget_msg = (
                        f"\x1b[33m  ╟─ Token budget exhausted "
                        f"({manager.total_tokens:,} >= {token_budget:,}); "
                        f"skipped {skipped} remaining agent(s)\x1b[0m"
                    )
                    if monitor is None:
                        print(budget_msg)
                break

    # ---- Stop the dashboard before printing phase summary -------------------
    if monitor is not None:
        monitor.stop()
        if budget_msg:
            print(budget_msg)

    phase1_elapsed = time.time() - start
    launched = len(results)
    note = f" ({skipped} skipped — budget)" if skipped else ""
    print(f"\x1b[35m  ╠═ Phase 1 done in {phase1_elapsed:.1f}s — "
          f"{launched} agent(s) ran{note} — handing to reviewer\x1b[0m")

    if not results:
        return ("(megaswarm: no agents ran — token budget already exhausted "
                "before any agent could start)")

    # ---- Phase 2: reviewer integrates / synthesises the outputs -----------
    label_to_task = dict(zip(labels, task_list))
    if divide:
        review_task = (
            "Multiple agents each completed a DISTINCT subtask of one larger goal. "
            "Integrate their results into ONE coherent report for the manager: "
            "confirm each subtask was done, flag conflicts or gaps between them, "
            "and note any follow-up needed.\n\n"
            f"=== OVERALL GOAL ===\n{task}\n\n"
            + "\n\n".join(
                f"=== {k.upper()} (subtask: {label_to_task.get(k, '?')[:120]}) ===\n{txt}"
                for k, txt in results.items())
        )
    else:
        review_task = (
            "Synthesise the following specialist outputs (all worked the SAME task) "
            "into ONE combined answer for the manager.\n\n"
            f"=== ORIGINAL TASK ===\n{task}\n\n"
            + "\n\n".join(f"=== {k.upper()} OUTPUT ===\n{txt}"
                          for k, txt in results.items())
        )

    reviewer_spec = SPECIALISTS["reviewer"]
    allowed = reviewer_spec["tools"]
    schemas = [s for s in TOOL_SCHEMAS if s["function"]["name"] in allowed]
    reviewer = Agent(
        root=manager.root,
        model=reviewer_spec["model"] or manager.model,
        auto_approve=manager.auto_approve,
        system_prompt=MEGASWARM_REVIEWER_PROMPT,
        tool_schemas=schemas,
        label="reviewer",
    )
    reviewer.client = manager.client

    print(f"\x1b[35m  ╟─ [reviewer] synthesising...\x1b[0m")
    phase2_start = time.time()
    synthesis = reviewer.run(review_task)
    phase2_elapsed = time.time() - phase2_start
    total_elapsed = time.time() - start

    manager.total_tokens += reviewer.total_tokens
    manager.delegation_count += 1

    print(f"\x1b[35m  ╟─ [reviewer] done in {phase2_elapsed:.1f}s\x1b[0m")
    print(f"\x1b[35;1m╚═ MEGASWARM complete in {total_elapsed:.1f}s\x1b[0m")

    return _clip_agent_result(
        synthesis or "(megaswarm reviewer returned no text)",
        "megaswarm synthesis",
    )


# ---------------------------------------------------------------------------
#  Pipeline — implement → review → fix (sequential), so bugs get caught
# ---------------------------------------------------------------------------

PIPELINE_REVIEWER_PROMPT = (
    "You are a strict code REVIEWER auditing a change a coder just made. You have "
    "read-only tools — inspect the actual files. Look for real bugs: logic errors, "
    "broken edge cases, security holes, things that won't run, and mismatches with "
    "the stated task.\n\n"
    "Respond in this exact format:\n"
    "VERDICT: CLEAN   (if the change is correct and complete)\n"
    "or\n"
    "VERDICT: ISSUES\n"
    "1. <file:line> <concrete problem and the fix>\n"
    "2. ...\n\n"
    "Only list issues you are confident are real. Do not nitpick style."
)

_PIPELINE_MAX_FIX_ROUNDS: int | None = None


def _get_pipeline_rounds() -> int:
    """Return the pipeline max fix rounds, initialising from env on first call."""
    global _PIPELINE_MAX_FIX_ROUNDS
    if _PIPELINE_MAX_FIX_ROUNDS is None:
        _PIPELINE_MAX_FIX_ROUNDS = int(os.environ.get("DEVBOT_PIPELINE_ROUNDS", "2"))
    return _PIPELINE_MAX_FIX_ROUNDS


def pipeline_schema() -> dict:
    """The `pipeline` tool: implement a task, then auto review-and-fix it."""
    return {
        "type": "function",
        "function": {
            "name": "pipeline",
            "description": (
                "Implement a coding task with an automatic review→fix loop: a coder "
                "makes the change, a reviewer audits the actual files for real bugs, "
                "and the coder fixes any findings (repeating until clean or the round "
                "limit). Prefer this over a plain `delegate(coder, ...)` for anything "
                "that must be correct — it catches bugs before they ship."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "A clear, self-contained coding task to implement.",
                    },
                },
                "required": ["task"],
            },
        },
    }


def run_pipeline(manager: "Agent", task: str) -> str:
    """Coder implements → reviewer audits → coder fixes, looping until clean."""
    print(f"\n\x1b[36;1m╔═ PIPELINE\x1b[0m \x1b[90m{task[:100]}\x1b[0m")

    transcript = [f"## Task\n{task}"]
    build = run_specialist(manager, "coder", task)
    transcript.append(f"## Initial implementation\n{build}")

    for round_no in range(1, _get_pipeline_rounds() + 1):
        review_task = (
            f"A coder was asked to do this task:\n{task}\n\n"
            f"Their summary of what they changed:\n{build}\n\n"
            "Inspect the actual files and audit the change now."
        )
        review = run_specialist_with_prompt(
            manager, "reviewer", PIPELINE_REVIEWER_PROMPT, review_task,
            label=f"reviewer-{round_no}")
        transcript.append(f"## Review (round {round_no})\n{review}")

        if "VERDICT: CLEAN" in review.upper() or "VERDICT: ISSUES" not in review.upper():
            print(f"\x1b[36m  ╟─ review round {round_no}: clean\x1b[0m")
            break

        print(f"\x1b[36m  ╟─ review round {round_no}: issues found — fixing\x1b[0m")
        fix_task = (
            f"A reviewer audited your change for this task:\n{task}\n\n"
            f"Fix every issue they raised:\n{review}\n\n"
            "Make the edits and briefly summarise what you changed."
        )
        build = run_specialist(manager, "coder", fix_task)
        transcript.append(f"## Fixes (round {round_no})\n{build}")
    else:
        print(f"\x1b[33m  ╟─ pipeline hit round limit "
              f"({_get_pipeline_rounds()}); some issues may remain\x1b[0m")

    print(f"\x1b[36;1m╚═ PIPELINE complete\x1b[0m")
    return _clip_agent_result("\n\n".join(transcript), "pipeline transcript")
