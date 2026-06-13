"""Tests for devbot.evolve_planner — the planner + critic for auto-evolve."""

import pytest

from devbot.evolve_planner import generate_plan, critique_plan


# ============================================================================
# _FakeAgent — mimics Agent.run() returning a predetermined response
# ============================================================================

class _FakeAgent:
    """An Agent stub whose ``run()`` returns a fixed string.

    Stores every call so tests can assert on prompts passed to the manager.
    """

    def __init__(self, response: str = ""):
        self.response = response
        self.runs: list[str] = []

    def run(self, prompt: str) -> str:
        self.runs.append(prompt)
        return self.response


# ============================================================================
# TestGeneratePlan
# ============================================================================

class TestGeneratePlan:
    """Tests for generate_plan(manager, context)."""

    VALID_PLAN_OUTPUT = """\
## Phase 1 — Add input validation
Validate all user inputs in cli.py to prevent crashes on malformed data.
This improves correctness and safety.

## Phase 2 — Fix race condition in session.py
Add a lock around shared state to prevent data corruption under parallel
access. This is a correctness fix.
"""

    def test_proposes_phases(self):
        """Mock the manager to return valid phase headings; verify phases."""
        manager = _FakeAgent(response=self.VALID_PLAN_OUTPUT)
        context = {"readme": "# My Project\nSome README content."}

        phases = generate_plan(manager, context)

        assert len(phases) == 2
        assert phases[0]["title"] == "Phase 1 — Add input validation"
        assert "input validation" in phases[0]["body"]
        assert phases[1]["title"] == "Phase 2 — Fix race condition in session.py"
        assert "race condition" in phases[1]["body"]

        # Manager.run() must have been called exactly once.
        assert len(manager.runs) == 1

    def test_no_phases_returns_empty(self):
        """Model returns text with no phase headings → []."""
        manager = _FakeAgent(response="This is just some random text.\n\nNo phases here.")
        context = {"readme": "stuff"}

        phases = generate_plan(manager, context)
        assert phases == []

    def test_malformed_output_returns_empty(self):
        """Manager.run() raises an exception → [] (no crash)."""

        class _FailingAgent:
            def run(self, prompt):
                raise RuntimeError("API exploded")

        manager = _FailingAgent()
        phases = generate_plan(manager, {"outline": "x"})
        assert phases == []

    def test_context_passed_in_prompt(self):
        """Verify the prompt string includes context values."""
        manager = _FakeAgent(response=self.VALID_PLAN_OUTPUT)
        context = {
            "readme": "README: This is a test project.",
            "outline": "OUTLINE: src/app.py, tests/",
            "tree": "TREE: .\n├── app.py",
        }

        generate_plan(manager, context)

        prompt = manager.runs[0]
        assert "README: This is a test project." in prompt
        assert "OUTLINE: src/app.py, tests/" in prompt
        assert "TREE: .\n├── app.py" in prompt
        # Context keys should appear as section headers
        assert "=== readme ===" in prompt
        assert "=== outline ===" in prompt
        assert "=== tree ===" in prompt

    def test_empty_context_keys_skipped(self):
        """Empty or None context values are not included in the prompt."""
        manager = _FakeAgent(response=self.VALID_PLAN_OUTPUT)
        context = {
            "readme": "stuff",
            "summary": "",      # empty → skip
            "outline": None,    # None → skip
            "tree": "tree here",
        }

        generate_plan(manager, context)

        prompt = manager.runs[0]
        assert "=== readme ===" in prompt
        assert "=== tree ===" in prompt
        assert "=== summary ===" not in prompt
        assert "=== outline ===" not in prompt

    def test_manager_run_returns_none(self):
        """If manager.run() returns None, we still handle it (treated as empty)."""

        class _NoneAgent:
            def run(self, prompt):
                return None

        phases = generate_plan(_NoneAgent(), {"readme": "x"})
        assert phases == []


# ============================================================================
# TestCritiquePlan
# ============================================================================

class TestCritiquePlan:
    """Tests for critique_plan(manager, phases)."""

    PHASES_INPUT = [
        {
            "title": "Phase 1 — Add input validation",
            "body": "Validate all user inputs in cli.py to prevent crashes.",
        },
        {
            "title": "Phase 2 — Rename variables to camelCase",
            "body": "Rename all snake_case locals to camelCase for consistency.",
        },
        {
            "title": "Phase 3 — Add timeout to network calls",
            "body": "Add httpx timeout to prevent hanging connections.",
        },
    ]

    CRITIQUE_KEEPS_1_AND_3 = """\
## Phase 1 — Add input validation
Justification: Improves correctness and safety by preventing crashes from malformed input. Score: 9.
Validate all user inputs in cli.py to prevent crashes.

## Phase 3 — Add timeout to network calls
Justification: Prevents the agent from hanging indefinitely on network issues. Score: 8.
Add httpx timeout to prevent hanging connections.
"""

    def test_keeps_good_phases(self):
        """Critic keeps 2 of 3 phases; verify returned list and keys."""
        manager = _FakeAgent(response=self.CRITIQUE_KEEPS_1_AND_3)
        result = critique_plan(manager, self.PHASES_INPUT)

        assert len(result) == 2
        assert result[0]["title"] == "Phase 1 — Add input validation"
        assert result[0]["justification"] == (
            "Improves correctness and safety by preventing crashes from "
            "malformed input. Score: 9."
        )
        assert "input validation" in result[0]["body"]

        assert result[1]["title"] == "Phase 3 — Add timeout to network calls"
        assert result[1]["justification"] == (
            "Prevents the agent from hanging indefinitely on network "
            "issues. Score: 8."
        )
        assert "timeout" in result[1]["body"]

        # Manager should have been called once with all three phases in the prompt
        assert len(manager.runs) == 1
        prompt = manager.runs[0]
        assert "Phase 1 — Add input validation" in prompt
        assert "Phase 2 — Rename variables to camelCase" in prompt
        assert "Phase 3 — Add timeout to network calls" in prompt

    def test_drops_all_returns_empty(self):
        """Critic returns NONE → []."""
        manager = _FakeAgent(response="NONE")
        result = critique_plan(manager, self.PHASES_INPUT)
        assert result == []

    def test_drops_all_empty_string_returns_empty(self):
        """Critic returns empty/whitespace string → []."""
        manager = _FakeAgent(response="   \n  ")
        result = critique_plan(manager, self.PHASES_INPUT)
        assert result == []

    def test_drops_pure_refactors(self):
        """The refactor phase (Phase 2) is absent from the surviving list."""
        manager = _FakeAgent(response=self.CRITIQUE_KEEPS_1_AND_3)
        result = critique_plan(manager, self.PHASES_INPUT)

        titles = {ph["title"] for ph in result}
        assert "Phase 2 — Rename variables to camelCase" not in titles
        assert len(result) == 2

    def test_malformed_output_returns_empty(self):
        """Manager.run() raises an exception → [] (no crash)."""

        class _FailingAgent:
            def run(self, prompt):
                raise RuntimeError("API exploded")

        result = critique_plan(_FailingAgent(), self.PHASES_INPUT)
        assert result == []

    def test_empty_input_returns_empty(self):
        """Passing [] as phases returns [] without calling the manager."""
        manager = _FakeAgent(response="should not be called")
        result = critique_plan(manager, [])
        assert result == []
        assert len(manager.runs) == 0

    def test_critique_output_without_justification(self):
        """If a surviving phase has no Justification line, justification is ''."""
        output = """\
## Phase 1 — Add input validation
Validate all user inputs in cli.py to prevent crashes.
"""
        manager = _FakeAgent(response=output)
        phases = [self.PHASES_INPUT[0]]  # just Phase 1

        result = critique_plan(manager, phases)

        assert len(result) == 1
        assert result[0]["title"] == "Phase 1 — Add input validation"
        assert result[0]["justification"] == ""
        assert "input validation" in result[0]["body"]

    def test_justification_removed_from_body(self):
        """The Justification line is stripped from the returned body."""
        output = """\
## Phase 3 — Add timeout to network calls
Justification: Prevents hanging. Score: 8.
Add httpx timeout to prevent hanging connections.
"""
        manager = _FakeAgent(response=output)
        phases = [self.PHASES_INPUT[2]]  # Phase 3

        result = critique_plan(manager, phases)

        assert len(result) == 1
        assert result[0]["justification"] == "Prevents hanging. Score: 8."
        # Body should NOT contain the justification line
        assert "Justification:" not in result[0]["body"]
        # Body should still contain the original description
        assert "Add httpx timeout" in result[0]["body"]

    def test_parse_error_returns_empty(self):
        """If parse_phases raises (unlikely), we still return [] gracefully."""
        manager = _FakeAgent(response="## Not a Phase — just some heading\nbody")

        result = critique_plan(manager, self.PHASES_INPUT)
        # parse_phases would find 0 phases (heading doesn't match "Phase N" pattern)
        assert result == []


# ============================================================================
# TestIntegration
# ============================================================================

class TestIntegration:
    """End-to-end: generate a plan then critique it — all with mocks."""

    def test_generate_then_critique_flow(self):
        """Full flow: planner proposes, critic filters — verify end-to-end."""
        plan_output = """\
## Phase 1 — Add input validation
Validate all user inputs to stop crashes. Improves correctness.

## Phase 2 — Rename utils.py to helpers.py
Pure rename for consistency. No functional change.

## Phase 3 — Add retry logic to API calls
Add exponential backoff for transient failures. Improves reliability.
"""

        critique_output = """\
## Phase 1 — Add input validation
Justification: Concrete correctness improvement. Score: 8.
Validate all user inputs to stop crashes. Improves correctness.

## Phase 3 — Add retry logic to API calls
Justification: Improves reliability under real-world conditions. Score: 7.
Add exponential backoff for transient failures. Improves reliability.
"""

        context = {"readme": "# Test Project", "summary": "A small CLI tool."}

        # Step 1: generate plan
        planner = _FakeAgent(response=plan_output)
        proposed = generate_plan(planner, context)
        assert len(proposed) == 3
        assert proposed[1]["title"] == "Phase 2 — Rename utils.py to helpers.py"

        # Step 2: critique the proposed phases
        critic = _FakeAgent(response=critique_output)
        surviving = critique_plan(critic, proposed)
        assert len(surviving) == 2

        titles = {ph["title"] for ph in surviving}
        assert "Phase 1 — Add input validation" in titles
        assert "Phase 3 — Add retry logic to API calls" in titles
        # The refactor phase was dropped
        assert "Phase 2 — Rename utils.py to helpers.py" not in titles

        # Both survivors have justifications
        for ph in surviving:
            assert ph["justification"] != ""
            assert "Justification:" not in ph["body"]

    def test_planner_returns_empty_then_critique_returns_empty(self):
        """If the planner finds no phases, critique gets [] and returns []."""
        planner = _FakeAgent(response="Nothing to improve here.")
        proposed = generate_plan(planner, {"readme": "x"})
        assert proposed == []

        critic = _FakeAgent(response="should not be called")
        surviving = critique_plan(critic, proposed)
        assert surviving == []
        assert len(critic.runs) == 0
