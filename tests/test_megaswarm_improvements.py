"""Tests for the megaswarm improvements: divide-and-conquer + review→fix pipeline.

These mock the sub-agent runners so no real API calls / tokens are used.
"""
import devbot.swarm as swarm
from devbot.swarm import (
    _clip_agent_result,
    megadelegate_schema,
    pipeline_schema,
    run_megaswarm,
    run_pipeline,
)


# ---------------------------------------------------------------------------
#  Schemas
# ---------------------------------------------------------------------------
class TestSchemas:
    def test_megadelegate_exposes_subtasks(self):
        props = megadelegate_schema()["function"]["parameters"]["properties"]
        assert "subtasks" in props
        assert props["subtasks"]["type"] == "array"
        assert {"task", "n", "role"} <= set(props)

    def test_pipeline_schema_shape(self):
        fn = pipeline_schema()["function"]
        assert fn["name"] == "pipeline"
        assert fn["parameters"]["required"] == ["task"]


# ---------------------------------------------------------------------------
#  run_megaswarm input validation (no API — early returns before manager use)
# ---------------------------------------------------------------------------
class TestMegaswarmValidation:
    def test_empty_subtasks_returns_error(self):
        out = run_megaswarm(None, "goal", subtasks=[])
        assert "empty" in out.lower()

    def test_whitespace_only_subtasks_returns_error(self):
        out = run_megaswarm(None, "goal", subtasks=["", "   ", "\n"])
        assert "empty" in out.lower()

    def test_unknown_role_returns_error(self):
        out = run_megaswarm(None, "goal", role="wizard")
        assert "unknown specialist" in out.lower()


class TestTokenEfficiency:
    def test_specialist_result_clip_is_configurable(self, monkeypatch):
        monkeypatch.setenv("DEVBOT_SPECIALIST_RESULT_LIMIT", "500")
        out = _clip_agent_result("x" * 650, "coder")
        assert len(out) < 650
        assert "coder truncated" in out


# ---------------------------------------------------------------------------
#  run_pipeline loop logic (mock the runners)
# ---------------------------------------------------------------------------
class _FakeManager:
    total_tokens = 0
    delegation_count = 0


class TestPipelineLoop:
    def test_stops_when_review_clean(self, monkeypatch):
        calls = []

        def fake_specialist(manager, role, task):
            calls.append(("build", role))
            return "implemented X"

        def fake_review(manager, role, prompt, task, label=None):
            calls.append(("review", label))
            return "VERDICT: CLEAN"

        monkeypatch.setattr(swarm, "run_specialist", fake_specialist)
        monkeypatch.setattr(swarm, "run_specialist_with_prompt", fake_review)

        out = run_pipeline(_FakeManager(), "do X")
        # one build, one review, no fix round
        assert [c[0] for c in calls] == ["build", "review"]
        assert "implemented X" in out

    def test_fixes_then_clean(self, monkeypatch):
        calls = []
        reviews = iter(["VERDICT: ISSUES\n1. bug here", "VERDICT: CLEAN"])

        def fake_specialist(manager, role, task):
            calls.append("build")
            return "did work"

        def fake_review(manager, role, prompt, task, label=None):
            calls.append("review")
            return next(reviews)

        monkeypatch.setattr(swarm, "run_specialist", fake_specialist)
        monkeypatch.setattr(swarm, "run_specialist_with_prompt", fake_review)

        out = run_pipeline(_FakeManager(), "do Y")
        # build, review(issues), fix-build, review(clean)
        assert calls == ["build", "review", "build", "review"]
        assert "Fixes" in out

    def test_respects_round_limit(self, monkeypatch):
        monkeypatch.setattr(swarm, "_PIPELINE_MAX_FIX_ROUNDS", 2)
        calls = []

        def fake_specialist(manager, role, task):
            calls.append("build")
            return "work"

        def fake_review(manager, role, prompt, task, label=None):
            calls.append("review")
            return "VERDICT: ISSUES\n1. still broken"  # never clean

        monkeypatch.setattr(swarm, "run_specialist", fake_specialist)
        monkeypatch.setattr(swarm, "run_specialist_with_prompt", fake_review)

        run_pipeline(_FakeManager(), "do Z")
        # initial build + 2 rounds of (review, fix-build) = 1 + 2 reviews + 2 builds
        assert calls.count("review") == 2
        assert calls.count("build") == 3
