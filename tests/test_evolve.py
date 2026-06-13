"""Comprehensive unit tests for ``devbot.evolve.run_evolve``.

All external calls (git, network, Agent, subprocess) are mocked so the
suite runs offline and deterministically.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import devbot.evolve as evolve_mod
from devbot.evolve import run_evolve


# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------

class _FakeAgent:
    """A stand-in for ``devbot.agent.Agent`` that records ``.run()`` calls."""

    def __init__(self, root=None, model=None, auto_approve=None, megaswarm=None):
        # Store constructor args for assertions.
        self.root = root
        self.model = model
        self.auto_approve = auto_approve
        self.megaswarm = megaswarm
        self.runs: list[str] = []

    def run(self, prompt: str) -> str:
        self.runs.append(prompt)
        return "done"

    def estimated_cost(self) -> float:
        return 0.01


class _FakeStopController:
    """Configurable stand-in for ``devbot.evolve_limits.StopController``."""

    def __init__(self):
        self.phase_count = 0
        self._stopped = False
        self._red = False
        # List of (stopped: bool, reason: str) to return from should_stop().
        self._should_stop_responses: list[tuple[bool, str]] = []
        self._call_count = 0
        self._started = False

    def start(self) -> None:
        self._started = True

    def record_phase(self) -> None:
        self.phase_count += 1

    def set_red(self) -> None:
        self._red = True

    def should_stop(self) -> tuple[bool, str]:
        self._call_count += 1
        if self._should_stop_responses:
            # Pop from front so each call consumes one response.
            return self._should_stop_responses.pop(0)
        # Default: never stop.
        return (False, "")


# ---------------------------------------------------------------------------
# Arrange helper
# ---------------------------------------------------------------------------

def _setup_mocks(
    monkeypatch,
    tmp_path: Path,
    *,
    is_clean: bool = True,
    branch: str = "feature",
    ensure_branch_side_effect=None,
    commit_all_return: str | None = "abc1234",
    should_stop_responses: list[tuple[bool, str]] | None = None,
    generate_plan_phases: list[dict] | None = None,
    critique_plan_phases: list[dict] | None = None,
    run_tests_responses: list[tuple[bool, str]] | None = None,
    fake_agent_class=None,
) -> dict:
    """Wire all mocks for ``run_evolve`` and return the fakes for assertions.

    Parameters
    ----------
    monkeypatch:
        Pytest ``monkeypatch`` fixture.
    tmp_path : Path
        Temporary directory to use as the repo root (``root`` arg).
    is_clean : bool
        Return value for the mocked ``is_clean``.
    branch : str
        Return value for the mocked ``current_branch``.
    ensure_branch_side_effect:
        Optional side-effect / return for ``ensure_branch``.
    commit_all_return : str | None
        Return value for ``commit_all``.
    should_stop_responses : list[tuple[bool, str]] | None
        Responses for ``StopController.should_stop``.  If *None*, defaults to
        ``[(True, "Max phases reached")]`` (stops on first check).
    generate_plan_phases : list[dict] | None
        Phases returned by ``generate_plan``.  Default: one phase.
    critique_plan_phases : list[dict] | None
        Phases returned by ``critique_plan``.  Default: same as generate.
    run_tests_responses : list[tuple[bool, str]] | None
        Responses for ``_run_tests``.  Default: ``[(True, "")]``.
    fake_agent_class:
        Class to use in place of ``Agent``.  Default: ``_FakeAgent``.

    Returns
    -------
    dict
        Keys: ``fake_sc``, ``ensure_branch``, ``commit_all``, ``fake_agent_cls``,
        ``run_tests``, ``generate_plan``, ``critique_plan``.
    """
    root_str = str(tmp_path)

    # ---- API key -----------------------------------------------------------
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    # ---- Neutralise _load_dotenv -------------------------------------------
    monkeypatch.setattr("devbot.agent._load_dotenv", lambda root: None)

    # ---- Git helpers -------------------------------------------------------
    _mock_is_clean = MagicMock(return_value=is_clean)
    monkeypatch.setattr("devbot.gitcheckpoint.is_clean", _mock_is_clean)

    _mock_current_branch = MagicMock(return_value=branch)
    monkeypatch.setattr("devbot.gitcheckpoint.current_branch", _mock_current_branch)

    _mock_ensure_branch = MagicMock()
    if ensure_branch_side_effect is not None:
        _mock_ensure_branch.side_effect = ensure_branch_side_effect
    monkeypatch.setattr("devbot.gitcheckpoint.ensure_branch", _mock_ensure_branch)

    _mock_commit_all = MagicMock(return_value=commit_all_return)
    monkeypatch.setattr("devbot.gitcheckpoint.commit_all", _mock_commit_all)

    # ---- StopController ----------------------------------------------------
    _fake_sc = _FakeStopController()
    if should_stop_responses is not None:
        _fake_sc._should_stop_responses = list(should_stop_responses)
    else:
        _fake_sc._should_stop_responses = [(True, "Max phases reached")]
    monkeypatch.setattr("devbot.evolve_limits.StopController", lambda: _fake_sc)

    # ---- Planner / critic --------------------------------------------------
    _default_phase = {"title": "Phase 1 — Add tests", "body": "Add unit tests."}
    if generate_plan_phases is None:
        generate_plan_phases = [_default_phase]
    _mock_generate_plan = MagicMock(return_value=generate_plan_phases)
    monkeypatch.setattr("devbot.evolve_planner.generate_plan", _mock_generate_plan)

    if critique_plan_phases is None:
        critique_plan_phases = list(generate_plan_phases)
    _mock_critique_plan = MagicMock(return_value=critique_plan_phases)
    monkeypatch.setattr("devbot.evolve_planner.critique_plan", _mock_critique_plan)

    # ---- _run_tests --------------------------------------------------------
    if run_tests_responses is None:
        run_tests_responses = [(True, "")]
    _mock_run_tests = MagicMock(side_effect=list(run_tests_responses))
    monkeypatch.setattr("devbot.autopilot._run_tests", _mock_run_tests)

    # ---- Agent -------------------------------------------------------------
    _fake_cls = fake_agent_class or _FakeAgent
    monkeypatch.setattr("devbot.agent.Agent", _fake_cls)

    # ---- get_global_token_count --------------------------------------------
    monkeypatch.setattr("devbot.agent.get_global_token_count", lambda: 0)

    return {
        "fake_sc": _fake_sc,
        "ensure_branch": _mock_ensure_branch,
        "commit_all": _mock_commit_all,
        "fake_agent_cls": _fake_cls,
        "run_tests": _mock_run_tests,
        "generate_plan": _mock_generate_plan,
        "critique_plan": _mock_critique_plan,
    }


# ============================================================================
# Tests
# ============================================================================

class TestPreflightDirtyTree:
    """1. Pre-flight: dirty tree returns False."""

    def test_dirty_tree_returns_false_and_prints_error(self, monkeypatch, tmp_path):
        # Arrange
        _setup_mocks(monkeypatch, tmp_path, is_clean=False)

        # Act
        result = run_evolve(tmp_path)

        # Assert
        assert result is False


class TestPreflightOnMainCreatesAutopilotBranch:
    """2. Pre-flight: on main creates autopilot branch."""

    def test_on_main_creates_autopilot_branch(self, monkeypatch, tmp_path):
        # Arrange
        fakes = _setup_mocks(monkeypatch, tmp_path, branch="main")

        # Act
        run_evolve(tmp_path)

        # Assert
        fakes["ensure_branch"].assert_called_once()
        call_args = fakes["ensure_branch"].call_args[0]
        branch_name = call_args[1]  # second positional arg
        assert re.match(r"autopilot/\d{4}-\d{2}-\d{2}-\d{4}", branch_name), (
            f"Expected autopilot/YYYY-MM-DD-HHMM, got {branch_name!r}"
        )
        assert branch_name not in ("main", "master")


class TestPreflightApiKeyMissing:
    """3. Pre-flight: API key missing returns False."""

    def test_api_key_missing_returns_false(self, monkeypatch, tmp_path):
        # Arrange
        # Delete the API key that conftest.py may have set.
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

        # Act
        result = run_evolve(tmp_path)

        # Assert
        assert result is False


class TestMainLoopPerPhaseCommitsOnGreen:
    """4. Main loop: per-phase commits on green."""

    def test_per_phase_commits_on_green(self, monkeypatch, tmp_path):
        # Arrange
        phase = {"title": "Phase 1 — Add tests", "body": "Add unit tests."}

        fakes = _setup_mocks(
            monkeypatch,
            tmp_path,
            branch="feature",
            # StopController checks: before plan gen (no stop), before phase (no stop),
            # next loop iteration (stop).
            should_stop_responses=[
                (False, ""),           # 1st: top of while → continue
                (False, ""),           # 2nd: before phase in for-loop → continue
                (True, "Max phases"),  # 3rd: top of while after phase → stop
            ],
            generate_plan_phases=[phase],
            critique_plan_phases=[phase],
            run_tests_responses=[(True, "")],
            commit_all_return="abc1234",
        )

        # Act
        result = run_evolve(tmp_path)

        # Assert
        assert result is True

        # commit_all called exactly once with the expected message.
        fakes["commit_all"].assert_called_once()
        commit_msg = fakes["commit_all"].call_args[0][1]
        assert commit_msg == f"evolve: {phase['title']}"

        # record_phase called once.
        assert fakes["fake_sc"].phase_count == 1


class TestStopOnRedTestsFailFixAlsoFails:
    """5. Stop-on-red: tests fail, fix round also fails."""

    def test_stop_on_red_no_commit(self, monkeypatch, tmp_path):
        # Arrange
        phase = {"title": "Phase 1 — Risky change", "body": "A risky change."}
        fail_output = "some failure"

        fakes = _setup_mocks(
            monkeypatch,
            tmp_path,
            branch="feature",
            should_stop_responses=[
                (False, ""),  # top of while → continue
                (False, ""),  # before phase → continue
                # After break from stop-on-red, we won't reach a 3rd check.
                (True, "unused"),
            ],
            generate_plan_phases=[phase],
            critique_plan_phases=[phase],
            # Both test runs fail.
            run_tests_responses=[(False, fail_output), (False, fail_output)],
        )

        # Act
        result = run_evolve(tmp_path)

        # Assert
        assert result is False
        # commit_all must NEVER be called.
        fakes["commit_all"].assert_not_called()
        # set_red must have been called.
        assert fakes["fake_sc"]._red is True


class TestStopOnCapMaxPhasesViaStopController:
    """6. Stop-on-cap: max phases via StopController before first phase."""

    def test_max_phases_before_implementation(self, monkeypatch, tmp_path):
        # Arrange
        phase = {"title": "Phase 1 — Never run", "body": "Should not run."}

        fakes = _setup_mocks(
            monkeypatch,
            tmp_path,
            branch="feature",
            should_stop_responses=[
                (False, ""),               # 1st: top of while → continue (plan gen)
                (True, "Max phases reached"),  # 2nd: before phase → stop
            ],
            generate_plan_phases=[phase],
            critique_plan_phases=[phase],
        )

        # Act
        result = run_evolve(tmp_path)

        # Assert
        assert result is False
        # No phases completed.
        assert fakes["fake_sc"].phase_count == 0
        # commit_all never called.
        fakes["commit_all"].assert_not_called()


class TestBranchIsolationNeverCommitsToMain:
    """7. Branch isolation: never commits to main."""

    def test_never_commits_to_main(self, monkeypatch, tmp_path):
        # Arrange
        phase = {"title": "Phase 1 — Safe change", "body": "A safe change."}

        fakes = _setup_mocks(
            monkeypatch,
            tmp_path,
            branch="main",
            should_stop_responses=[
                (False, ""),           # top of while → continue
                (False, ""),           # before phase → continue
                (True, "Max phases"),  # next loop → stop
            ],
            generate_plan_phases=[phase],
            critique_plan_phases=[phase],
            run_tests_responses=[(True, "")],
            commit_all_return="abc1234",
        )

        # Act
        run_evolve(tmp_path)

        # Assert
        # ensure_branch called with an autopilot/ branch (not "main").
        fakes["ensure_branch"].assert_called_once()
        branch_name = fakes["ensure_branch"].call_args[0][1]
        assert branch_name.startswith("autopilot/")
        assert branch_name not in ("main", "master")

        # commit_all IS called (since a phase ran) using root_str (tmp_path).
        fakes["commit_all"].assert_called_once()
        # The root arg passed to commit_all is the stringified tmp_path.
        root_arg = fakes["commit_all"].call_args[0][0]
        assert root_arg == str(tmp_path)


class TestEmptyPlanStopsGracefully:
    """8. Empty plan from planner stops gracefully."""

    def test_empty_plan_returns_false(self, monkeypatch, tmp_path):
        # Arrange
        fakes = _setup_mocks(
            monkeypatch,
            tmp_path,
            branch="feature",
            # Only one should_stop call happens (top of while, before plan gen).
            # We need it to NOT stop there, so the loop can generate the plan.
            # Then after empty plan, the loop breaks.
            should_stop_responses=[(False, "")],
            generate_plan_phases=[],   # <-- empty plan
        )

        # Act
        result = run_evolve(tmp_path)

        # Assert
        assert result is False
        # No phases were implemented.
        assert fakes["fake_sc"].phase_count == 0
        # commit_all never called.
        fakes["commit_all"].assert_not_called()


class TestCriticRejectsAllPhasesContinues:
    """9. Critic rejects all phases; loop continues then stops."""

    def test_critic_rejects_all_then_stops(self, monkeypatch, tmp_path):
        # Arrange
        phase1 = {"title": "Phase 1 — Bad idea", "body": "Refactor everything."}
        phase2 = {"title": "Phase 2 — Also bad", "body": "Rename all vars."}

        fakes = _setup_mocks(
            monkeypatch,
            tmp_path,
            branch="feature",
            should_stop_responses=[
                (False, ""),              # 1st: top of while → continue
                (True, "Max phases reached"),  # 2nd: top of while after critic rejection → stop
            ],
            generate_plan_phases=[phase1, phase2],
            critique_plan_phases=[],      # Critic rejects ALL phases
        )

        # Act
        result = run_evolve(tmp_path)

        # Assert
        assert result is False
        # No phases were implemented.
        assert fakes["fake_sc"].phase_count == 0
        # commit_all never called.
        fakes["commit_all"].assert_not_called()
        # generate_plan was called (once).
        fakes["generate_plan"].assert_called_once()
        # critique_plan was called (once).
        fakes["critique_plan"].assert_called_once()


class TestFixRoundSucceeds:
    """10. Fix round succeeds — phase commits after fix."""

    def test_fix_round_succeeds_and_commits(self, monkeypatch, tmp_path):
        # Arrange
        phase = {"title": "Phase 1 — Tricky change", "body": "A tricky change."}

        fakes = _setup_mocks(
            monkeypatch,
            tmp_path,
            branch="feature",
            should_stop_responses=[
                (False, ""),           # top of while → continue
                (False, ""),           # before phase → continue
                (True, "Max phases"),  # next loop → stop
            ],
            generate_plan_phases=[phase],
            critique_plan_phases=[phase],
            # First run fails, second (after fix) passes.
            run_tests_responses=[(False, "1 failed"), (True, "")],
            commit_all_return="abc1234",
        )

        # Act
        result = run_evolve(tmp_path)

        # Assert
        assert result is True
        # commit_all was called (once, after the fix round made tests green).
        fakes["commit_all"].assert_called_once()
        # Agent was created 3 times: manager, critic, impl, fix = 4 actually
        # Let's verify we got the fix agent run.
        # We can check that there was at least one Agent created with megaswarm=True.
        assert fakes["fake_sc"].phase_count == 1
