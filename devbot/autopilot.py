"""Autopilot: run a plan.md end-to-end, one phase at a time, unattended.

Each `## Phase N ...` section becomes its own megaswarm run with a FRESH agent
(so context and cost stay bounded per phase). After a phase, the test suite is
run: if it fails, one fix round is attempted; if it still fails, the run stops
so a human can inspect. Nothing is committed — all changes are left in the
working tree for review.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Any level-2 heading; we treat those titled "Phase N ..." as phases.
HEADING_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)
_PHASE_TITLE_RE = re.compile(r"Phase\s+\d+\b")

# Imported lazily-overridable for testing.
from .agent import Agent, check_global_budget_exceeded, get_global_token_count, _get_global_budget


def parse_phases(plan_text: str) -> list[dict]:
    """Split plan text into [{title, body}] for each '## Phase N ...' heading.

    A phase's body runs until the next '##' heading of ANY kind, so trailing
    non-phase sections (e.g. "## Cross-cutting notes") aren't absorbed.
    """
    headings = list(HEADING_RE.finditer(plan_text))
    phases = []
    for i, m in enumerate(headings):
        title = m.group(1).strip()
        if not _PHASE_TITLE_RE.match(title):
            continue
        start = m.start()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(plan_text)
        phases.append({"title": title, "body": plan_text[start:end].strip()})
    return phases


def _run_tests(root: Path) -> tuple[bool, str]:
    """Run the project's pytest suite. Returns (passed, tail_of_output)."""
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=root, capture_output=True, text=True,
            timeout=900, encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        return False, "pytest not installed (pip install -e .[dev])"
    except subprocess.TimeoutExpired:
        return False, "pytest timed out after 900s"
    out = (r.stdout or "") + (r.stderr or "")
    return r.returncode == 0, out[-3000:]


def run_plan(root: Path, plan_path: str = "plan.md", model: str | None = None,
             max_phases: int = 20) -> bool:
    """Implement every phase in *plan_path* sequentially. Returns True if all
    phases completed with green tests, False if it stopped early."""
    plan_file = root / plan_path
    if not plan_file.is_file():
        print(f"\x1b[31m[autopilot] {plan_path} not found in {root}\x1b[0m")
        return False

    plan_text = plan_file.read_text(encoding="utf-8")
    phases = parse_phases(plan_text)
    if not phases:
        print("\x1b[31m[autopilot] no '## Phase N' sections found in the plan\x1b[0m")
        return False
    if len(phases) > max_phases:
        print(f"\x1b[33m[autopilot] {len(phases)} phases found; capping at "
              f"{max_phases}\x1b[0m")
        phases = phases[:max_phases]

    print(f"\x1b[36;1m[autopilot] running {len(phases)} phase(s) from {plan_path}\x1b[0m")

    def _agent() -> Agent:
        return Agent(root=root, model=model, auto_approve=True, megaswarm=True)

    for idx, ph in enumerate(phases, 1):
        if check_global_budget_exceeded():
            print(f"\x1b[31m[autopilot] Global token budget exhausted "
                  f"({get_global_token_count():,} >= {_get_global_budget():,}). "
                  f"{idx-1}/{len(phases)} phases completed.\x1b[0m")
            return False

        print(f"\n\x1b[36;1m{'='*70}\n[autopilot] PHASE {idx}/{len(phases)}: "
              f"{ph['title']}\n{'='*70}\x1b[0m")

        _agent().run(
            f"You are implementing one phase of a multi-phase plan.\n\n"
            f"=== FULL PLAN (for context) ===\n{plan_text}\n\n"
            f"=== IMPLEMENT ONLY THIS PHASE NOW ===\n{ph['body']}\n\n"
            "Use the `pipeline` tool for every code change (it enforces review). "
            "Do not start other phases. When finished, make sure the test suite "
            "passes."
        )

        ok, out = _run_tests(root)
        if not ok:
            print(f"\x1b[33m[autopilot] tests failing after {ph['title']} — "
                  f"one fix round\x1b[0m")
            _agent().run(
                f"After implementing this phase, the test suite is FAILING:\n\n"
                f"{ph['body']}\n\n=== PYTEST OUTPUT ===\n{out}\n\n"
                "Diagnose and fix the failure using the `pipeline` tool, then make "
                "sure `pytest -q` passes. Do not start any other phase."
            )
            ok, out = _run_tests(root)

        if not ok:
            print(f"\x1b[31m[autopilot] STOPPING: {ph['title']} still failing after "
                  f"a fix round. Changes left in the working tree.\x1b[0m")
            print(out)
            return False

        print(f"\x1b[32m[autopilot] {ph['title']} complete — tests green\x1b[0m")

    print("\n\x1b[32;1m[autopilot] all phases complete. Changes are UNCOMMITTED "
          "in the working tree — review, then commit.\x1b[0m")
    return True
