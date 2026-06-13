"""Tests for the autopilot plan runner. No network calls (Agent is mocked)."""

import devbot.autopilot as autopilot
from devbot.autopilot import parse_phases, run_plan


PLAN = """\
# Plan

Intro text that is not a phase.

## Phase 1 — First thing
Do the first thing.
- detail a

## Phase 2: Second thing
Do the second thing.

## Cross-cutting notes
Not a phase.
"""


# ---------------------------------------------------------------------------
#  parse_phases
# ---------------------------------------------------------------------------
class TestParsePhases:
    def test_finds_phase_headings(self):
        phases = parse_phases(PLAN)
        assert len(phases) == 2
        assert phases[0]["title"] == "Phase 1 — First thing"
        assert phases[1]["title"] == "Phase 2: Second thing"

    def test_body_includes_content_until_next_section(self):
        phases = parse_phases(PLAN)
        assert "Do the first thing." in phases[0]["body"]
        assert "detail a" in phases[0]["body"]
        # must not bleed into the next phase or trailing non-phase section
        assert "Second thing" not in phases[0]["body"]
        assert "Cross-cutting" not in phases[1]["body"]

    def test_no_phases_returns_empty(self):
        assert parse_phases("# Just a title\n\nsome prose") == []


# ---------------------------------------------------------------------------
#  run_plan control flow (mock Agent + _run_tests)
# ---------------------------------------------------------------------------
class _FakeAgent:
    instances = []

    def __init__(self, *a, **k):
        self.runs = []
        _FakeAgent.instances.append(self)

    def run(self, prompt):
        self.runs.append(prompt)
        return "done"


def _setup(monkeypatch, tmp_path, test_results):
    """Wire fakes; test_results is a list of bools returned by _run_tests in order."""
    _FakeAgent.instances = []
    monkeypatch.setattr(autopilot, "Agent", _FakeAgent)
    results = iter(test_results)
    monkeypatch.setattr(autopilot, "_run_tests",
                        lambda root: (next(results), "pytest output"))
    (tmp_path / "plan.md").write_text(PLAN, encoding="utf-8")


class TestRunPlan:
    def test_all_phases_pass(self, tmp_path, monkeypatch):
        # phase1 green, phase2 green
        _setup(monkeypatch, tmp_path, [True, True])
        ok = run_plan(tmp_path)
        assert ok is True
        # one agent per phase, no fix rounds
        assert len(_FakeAgent.instances) == 2

    def test_fix_round_recovers(self, tmp_path, monkeypatch):
        # phase1: fail then (fix) pass; phase2: pass
        _setup(monkeypatch, tmp_path, [False, True, True])
        ok = run_plan(tmp_path)
        assert ok is True
        # phase1 build + phase1 fix + phase2 build = 3 agents
        assert len(_FakeAgent.instances) == 3

    def test_stops_when_fix_fails(self, tmp_path, monkeypatch):
        # phase1: fail, fix, still fail -> stop before phase2
        _setup(monkeypatch, tmp_path, [False, False])
        ok = run_plan(tmp_path)
        assert ok is False
        # phase1 build + phase1 fix only; never reached phase2
        assert len(_FakeAgent.instances) == 2

    def test_missing_plan_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(autopilot, "Agent", _FakeAgent)
        assert run_plan(tmp_path, "nope.md") is False

    def test_no_phases_returns_false(self, tmp_path, monkeypatch):
        _FakeAgent.instances = []
        monkeypatch.setattr(autopilot, "Agent", _FakeAgent)
        (tmp_path / "plan.md").write_text("# no phases here", encoding="utf-8")
        assert run_plan(tmp_path) is False
