"""Auto-evolve driver: self-evolving loop that plans, critiques, implements,
and commits improvement phases unattended.

Orchestrates a loop where an LLM planner proposes phases, a critic filters
them, and implementation agents build each surviving phase — verifying with
the test suite and committing on green.
"""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_tree(root: Path) -> str:
    """Return a file listing of the repo.

    Tries ``git ls-tree -r --name-only HEAD`` first; falls back to a
    simple ``Path.rglob`` listing (excluding ``.git`` and hidden dirs).
    """
    try:
        r = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass

    # Fallback: collect relative paths, skip .git and dunder dirs.
    lines: list[str] = []
    try:
        for p in sorted(root.rglob("*")):
            if p.is_dir():
                continue
            rel = p.relative_to(root)
            parts = rel.parts
            if any(part == ".git" or part.startswith("__pycache__")
                   for part in parts):
                continue
            lines.append(str(rel))
    except Exception:
        return ""
    return "\n".join(lines)


def _read_readme(root: Path) -> str:
    """Return the first 200 lines of README.md, or an empty string."""
    readme = root / "README.md"
    if not readme.is_file():
        return ""
    try:
        text = readme.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()[:200]
        return "\n".join(lines)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_evolve(root: Path, model: str | None = None, provider: str | None = None) -> bool:
    """Run the auto-evolve loop.

    Parameters
    ----------
    root : Path
        Project root (must be inside a git repository).
    model : str | None
        Model id to use for all agents (default: ``DEVBOT_MODEL`` env or
        the built-in default).
    provider : str | None
        Provider id to use for all agents, e.g. ``local-vibethinker``.

    Returns
    -------
    bool
        ``True`` if at least one phase was completed and committed,
        ``False`` otherwise.
    """
    # -- imports (lazy at function level to avoid circular imports at load) --
    from devbot.gitcheckpoint import (
        is_clean,
        current_branch,
        ensure_branch,
        commit_all,
    )
    from devbot.evolve_limits import StopController
    from devbot.evolve_planner import generate_plan, critique_plan
    from devbot.agent import Agent, get_global_token_count
    from devbot.autopilot import _run_tests

    # Pre-flight: load project config/.env so provider/key checks match Agent.
    from devbot.agent import _load_dotenv, get_llm_provider_settings
    from devbot.config import load_project_config, apply_project_config
    load_project_config(root)
    _load_dotenv(root)
    apply_project_config()
    provider_settings = get_llm_provider_settings(model, provider)
    if provider_settings.requires_deepseek_key and not provider_settings.api_key:
        print("[auto-evolve] DEEPSEEK_API_KEY is not set.  Please set it and "
              "try again, or set DEVBOT_PROVIDER=local-vibethinker.")
        return False

    root_str = str(root)

    # ---- 1. Pre-flight guards ------------------------------------------------
    try:
        if not is_clean(root_str):
            print("[auto-evolve] Working tree is NOT clean.  Please commit or "
                  "stash changes before running evolve.")
            return False
    except Exception as exc:
        print(f"[auto-evolve] git error checking clean status: {exc}")
        return False

    try:
        branch = current_branch(root_str)
    except Exception as exc:
        print(f"[auto-evolve] git error getting current branch: {exc}")
        return False

    branch_name: str = branch
    if branch in ("main", "master"):
        ts = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
        branch_name = f"autopilot/{ts}"
        try:
            ensure_branch(root_str, branch_name)
            print(f"[auto-evolve] Created/checked out branch: {branch_name}")
        except Exception as exc:
            print(f"[auto-evolve] Failed to create branch '{branch_name}': {exc}")
            return False

    # ---- 2. Setup ------------------------------------------------------------
    sc = StopController()
    sc.start()

    commit_shas: list[tuple[str, str]] = []
    total_phases_completed = 0
    cost_estimate = 0.0

    # ---- 3. Main loop --------------------------------------------------------
    stopped = False
    reason = ""

    while True:
        # Check stop conditions before generating a new plan.
        stopped, reason = sc.should_stop()
        if stopped:
            break

        # Build context for the planner.
        context = {
            "tree": _capture_tree(root),
            "readme": _read_readme(root),
        }

        # --- Generate plan ---
        try:
            manager = Agent(root=root, model=model, auto_approve=True,
                            megaswarm=True, provider=provider)
            proposed = generate_plan(manager, context)
            cost_estimate += manager.estimated_cost()
        except (Exception, SystemExit) as exc:
            print(f"[auto-evolve] Error generating plan: {exc}")
            continue

        if not proposed:
            print("[auto-evolve] Planner returned no phases; stopping.")
            break

        print(f"[auto-evolve] Planner proposed {len(proposed)} phase(s)")

        # --- Critique plan ---
        try:
            critic = Agent(root=root, model=model, auto_approve=True,
                           megaswarm=True, provider=provider)
            surviving = critique_plan(critic, proposed)
            cost_estimate += critic.estimated_cost()
        except (Exception, SystemExit) as exc:
            print(f"[auto-evolve] Error during critique: {exc}")
            continue

        if not surviving:
            print("[auto-evolve] Critic rejected all phases; generating next plan.")
            continue

        print(f"[auto-evolve] Critic accepted {len(surviving)} phase(s)")

        # --- Implement each surviving phase ---
        for phase in surviving:
            # Check stop before each phase.
            stopped, reason = sc.should_stop()
            if stopped:
                break

            title = phase["title"]
            body = phase["body"]
            print(f"\n[auto-evolve] Implementing phase: {title}")

            try:
                impl_agent = Agent(root=root, model=model, auto_approve=True,
                                   megaswarm=True, provider=provider)
                prompt = (
                    f"You are implementing one phase of a multi-phase plan.\n\n"
                    f"=== PHASE ===\n# {title}\n{body}\n\n"
                    "Use the `pipeline` tool for every code change (it enforces "
                    "review). When finished, make sure the test suite passes."
                )
                impl_agent.run(prompt)
                cost_estimate += impl_agent.estimated_cost()
            except (Exception, SystemExit) as exc:
                print(f"[auto-evolve] Error during implementation of "
                      f"'{title}': {exc}")
                sc.set_red()
                stopped = True
                reason = f"Implementation error in '{title}': {exc}"
                break

            # Run tests.
            try:
                ok, out = _run_tests(root)
            except Exception as exc:
                print(f"[auto-evolve] Error running tests after '{title}': {exc}")
                ok, out = False, str(exc)

            if not ok:
                # One fix round.
                print(f"[auto-evolve] Tests failing after '{title}' — "
                      f"one fix round")
                try:
                    fix_agent = Agent(root=root, model=model, auto_approve=True,
                                      megaswarm=True, provider=provider)
                    fix_agent.run(
                        f"After implementing this phase, the test suite is "
                        f"FAILING:\n\n{body}\n\n=== PYTEST OUTPUT ===\n{out}\n\n"
                        "Diagnose and fix the failure using the `pipeline` tool, "
                        "then make sure `pytest -q` passes."
                    )
                    cost_estimate += fix_agent.estimated_cost()
                except (Exception, SystemExit) as exc:
                    print(f"[auto-evolve] Error during fix round for "
                          f"'{title}': {exc}")

                try:
                    ok, out = _run_tests(root)
                except Exception as exc:
                    ok, out = False, str(exc)

            if not ok:
                # Stop-on-red.
                print(f"[auto-evolve] STOPPING: '{title}' still failing after "
                      f"fix round.")
                print(out[-2000:])
                sc.set_red()
                stopped = True
                reason = f"Tests still failing after '{title}'"
                break

            # Green — commit to the branch.
            print(f"[auto-evolve] '{title}' complete — tests green")
            try:
                sha = commit_all(root_str, f"evolve: {title}")
                if sha:
                    commit_shas.append((title, sha))
                    print(f"[auto-evolve]   committed {sha}")
                else:
                    print(f"[auto-evolve]   (nothing to commit)")
            except Exception as exc:
                print(f"[auto-evolve] Commit failed for '{title}': {exc}")
                # Don't stop the whole run for a commit failure — just note it.

            sc.record_phase()
            total_phases_completed += 1

        # After all phases in this plan, if stop was hit, break the outer loop.
        if stopped:
            break

    # ---- 4. Summary ----------------------------------------------------------
    print("\n=== AUTO-EVOLVE SUMMARY ===")
    print(f"Branch: {branch_name}")
    print(f"Phases completed: {total_phases_completed}")
    for title, sha in commit_shas:
        print(f"  {sha}  {title}")

    final_reason = reason if (stopped and reason) else (reason or "completed all plans")
    print(f"Stop reason: {final_reason}")
    print(f"Total tokens: {get_global_token_count():,}")
    print(f"Estimated cost: ${cost_estimate:.2f}")

    return total_phases_completed > 0
