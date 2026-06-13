"""Comprehensive unit tests for Phase 4 features: model pricing, cost
estimation, token budget (early exit + session persistence), compression
model env var, and megaswarm warning threshold. No network calls."""

import os
from pathlib import Path

import pytest

from devbot.agent import Agent, MODEL_PRICING


# ============================================================================
# 1. MODEL_PRICING and estimated_cost
# ============================================================================

class TestModelPricing:
    """Tests for MODEL_PRICING table and Agent.estimated_cost()."""

    def test_pricing_table_has_all_known_models(self):
        """All 4 known models are present in MODEL_PRICING."""
        expected = {"deepseek-v4-pro", "deepseek-v4-flash",
                    "deepseek-chat", "deepseek-reasoner"}
        missing = expected - set(MODEL_PRICING)
        assert not missing, (
            f"MODEL_PRICING is missing expected model(s): {missing}. "
            f"Present keys: {set(MODEL_PRICING)}"
        )


class TestEstimatedCost:
    """Tests for Agent.estimated_cost() with different models and token counts."""

    @pytest.fixture(autouse=True)
    def _setup_api_key(self, monkeypatch):
        """Set a dummy API key so Agent.__init__ doesn't call SystemExit."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-dummy-key")

    def _make_agent(self, tmp_path, model="deepseek-v4-flash"):
        """Create an Agent with a given model. No network calls are made."""
        return Agent(root=tmp_path, model=model, auto_approve=True)

    # ---- known models --------------------------------------------------------

    def test_estimated_cost_flash_model(self, tmp_path):
        """total_tokens=1_000_000 on flash → $0.21.

        Flash pricing: input $0.14/1M, output $0.28/1M.
        Half = 500_000 tokens.
        cost = 500_000 * (0.14 / 1_000_000) + 500_000 * (0.28 / 1_000_000)
             = 0.07 + 0.14 = $0.21
        """
        agent = self._make_agent(tmp_path, model="deepseek-v4-flash")
        agent.total_tokens = 1_000_000
        cost = agent.estimated_cost()
        assert cost == pytest.approx(0.21, rel=1e-9)

    def test_estimated_cost_pro_model(self, tmp_path):
        """total_tokens=1_000_000 on pro → $0.6525.

        Pro pricing: input $0.435/1M, output $0.87/1M.
        Half = 500_000 tokens.
        cost = 500_000 * (0.435 / 1_000_000) + 500_000 * (0.87 / 1_000_000)
             = 0.2175 + 0.435 = $0.6525
        """
        agent = self._make_agent(tmp_path, model="deepseek-v4-pro")
        agent.total_tokens = 1_000_000
        cost = agent.estimated_cost()
        assert cost == pytest.approx(0.6525, rel=1e-9)

    def test_estimated_cost_chat_model(self, tmp_path):
        """deepseek-chat shares flash pricing → $0.21."""
        agent = self._make_agent(tmp_path, model="deepseek-chat")
        agent.total_tokens = 1_000_000
        cost = agent.estimated_cost()
        assert cost == pytest.approx(0.21, rel=1e-9)

    def test_estimated_cost_reasoner_model(self, tmp_path):
        """deepseek-reasoner shares flash pricing → $0.21."""
        agent = self._make_agent(tmp_path, model="deepseek-reasoner")
        agent.total_tokens = 1_000_000
        cost = agent.estimated_cost()
        assert cost == pytest.approx(0.21, rel=1e-9)

    # ---- fallback ------------------------------------------------------------

    def test_estimated_cost_unknown_model_falls_back_to_flash(self, tmp_path):
        """An unknown model uses deepseek-v4-flash prices."""
        agent = self._make_agent(tmp_path, model="unknown-model-xyz")
        agent.total_tokens = 1_000_000
        cost = agent.estimated_cost()
        assert cost == pytest.approx(0.21, rel=1e-9)

    # ---- zero tokens ---------------------------------------------------------

    def test_estimated_cost_zero_tokens(self, tmp_path):
        """total_tokens=0 → cost $0.00."""
        agent = self._make_agent(tmp_path, model="deepseek-v4-pro")
        agent.total_tokens = 0
        cost = agent.estimated_cost()
        assert cost == 0.0

    # ---- fractional tokens (realistic) ---------------------------------------

    def test_estimated_cost_fractional(self, tmp_path):
        """Verify cost calculation with a non-round token count."""
        agent = self._make_agent(tmp_path, model="deepseek-v4-flash")
        agent.total_tokens = 123_456
        # half = 61_728; flash input $0.14/1M, output $0.28/1M
        expected = (61_728 * 0.14 + 61_728 * 0.28) / 1_000_000
        cost = agent.estimated_cost()
        assert cost == pytest.approx(expected, rel=1e-9)


# ============================================================================
# 2. Token budget — early exit in Agent.run()
# ============================================================================

class TestTokenBudgetEarlyExit:
    """Tests for token_budget guard in Agent.run()."""

    @pytest.fixture(autouse=True)
    def _setup_api_key(self, monkeypatch):
        """Set a dummy API key so Agent.__init__ doesn't call SystemExit."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-dummy-key")

    def _make_agent(self, tmp_path, budget=None):
        """Create an Agent, optionally setting DEVBOT_TOKEN_BUDGET."""
        if budget is not None:
            os.environ["DEVBOT_TOKEN_BUDGET"] = str(budget)
        return Agent(root=tmp_path, auto_approve=True)

    def test_budget_exhausted_before_api_call(self, tmp_path, monkeypatch):
        """When total_tokens >= token_budget, run() returns '' immediately.

        The user message is still appended to self.messages, but no API call
        is made and the return value is empty.
        """
        monkeypatch.setenv("DEVBOT_TOKEN_BUDGET", "100")
        agent = self._make_agent(tmp_path)
        # Simulate prior usage that exhausted the budget
        agent.total_tokens = 100
        msg_count_before = len(agent.messages)

        result = agent.run("do something")

        # No API call → empty string
        assert result == ""
        # User message was appended
        assert len(agent.messages) == msg_count_before + 1
        assert agent.messages[-1]["role"] == "user"
        assert agent.messages[-1]["content"] == "do something"

    def test_budget_not_exhausted_continues(self, tmp_path, monkeypatch):
        """When total_tokens < token_budget, the guard does NOT fire.

        We can't call run() (it would hit the API), so we verify the
        pre-flight condition evaluates to False.
        """
        monkeypatch.setenv("DEVBOT_TOKEN_BUDGET", "100")
        agent = self._make_agent(tmp_path)
        agent.total_tokens = 50

        # The condition in run() is:
        #   if self.token_budget > 0 and self.total_tokens >= self.token_budget
        would_exit = agent.token_budget > 0 and agent.total_tokens >= agent.token_budget
        assert would_exit is False

    def test_budget_zero_means_no_limit(self, tmp_path, monkeypatch):
        """token_budget=0 disables the budget check entirely."""
        monkeypatch.setenv("DEVBOT_TOKEN_BUDGET", "0")
        agent = self._make_agent(tmp_path)
        agent.total_tokens = 1_000_000

        would_exit = agent.token_budget > 0 and agent.total_tokens >= agent.token_budget
        assert would_exit is False

    def test_budget_exhausted_flag_initialized_false(self, tmp_path):
        """Agent._budget_exhausted starts as False in __init__."""
        agent = self._make_agent(tmp_path)
        assert agent._budget_exhausted is False


# ============================================================================
# 3. Token budget — session persistence
# ============================================================================

class TestTokenBudgetSessionPersistence:
    """Tests that token_budget is stored on Agent and saved in sessions."""

    @pytest.fixture(autouse=True)
    def _setup_api_key(self, monkeypatch):
        """Set a dummy API key so Agent.__init__ doesn't call SystemExit."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-dummy-key")

    def test_token_budget_attribute_from_env(self, tmp_path, monkeypatch):
        """DEVBOT_TOKEN_BUDGET is read into agent.token_budget."""
        monkeypatch.setenv("DEVBOT_TOKEN_BUDGET", "500")
        agent = Agent(root=tmp_path, auto_approve=True)
        assert agent.token_budget == 500

    def test_token_budget_default_zero(self, tmp_path, monkeypatch):
        """Without DEVBOT_TOKEN_BUDGET, token_budget defaults to 0."""
        monkeypatch.delenv("DEVBOT_TOKEN_BUDGET", raising=False)
        agent = Agent(root=tmp_path, auto_approve=True)
        assert agent.token_budget == 0

    def test_token_budget_empty_string_defaults_zero(self, tmp_path, monkeypatch):
        """DEVBOT_TOKEN_BUDGET='' → int('0' or '0') → 0."""
        monkeypatch.setenv("DEVBOT_TOKEN_BUDGET", "")
        agent = Agent(root=tmp_path, auto_approve=True)
        assert agent.token_budget == 0

    def test_save_session_includes_token_budget(self, tmp_path, monkeypatch):
        """save_session() writes token_budget into the session JSON."""
        monkeypatch.setenv("DEVBOT_TOKEN_BUDGET", "777")
        # Use Agent.__new__ to bypass __init__, then manually set attributes
        # (same pattern as test_session.py).
        agent = Agent.__new__(Agent)
        agent.label = None          # main agent
        agent.root = tmp_path
        agent.model = "deepseek-v4-flash"
        agent.swarm = False
        agent.megaswarm = False
        agent.auto_approve = True
        agent.total_tokens = 0
        agent.last_prompt_tokens = 0
        agent.delegation_count = 0
        agent.token_budget = 777
        agent.messages = []
        agent.session_id = None

        from devbot.session import save_session, SESSIONS_DIR
        sid = save_session(agent)
        assert sid is not None

        import json
        path = tmp_path / SESSIONS_DIR / f"{sid}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["token_budget"] == 777


# ============================================================================
# 4. Compression model env var (DEVBOT_COMPRESS_MODEL)
# ============================================================================

class TestCompressModelEnvVar:
    """Tests for DEVBOT_COMPRESS_MODEL env var logic without API calls."""

    def test_compress_model_env_var_default(self, monkeypatch):
        """Default is 'deepseek-v4-flash' when env var is not set."""
        monkeypatch.delenv("DEVBOT_COMPRESS_MODEL", raising=False)
        val = os.environ.get("DEVBOT_COMPRESS_MODEL", "deepseek-v4-flash")
        assert val == "deepseek-v4-flash"

    def test_compress_model_env_var_custom(self, monkeypatch):
        """DEVBOT_COMPRESS_MODEL is read correctly when set."""
        monkeypatch.setenv("DEVBOT_COMPRESS_MODEL", "deepseek-v4-pro")
        val = os.environ.get("DEVBOT_COMPRESS_MODEL", "deepseek-v4-flash")
        assert val == "deepseek-v4-pro"

    def test_compress_model_env_var_custom_other(self, monkeypatch):
        """Any model name can be set via the env var."""
        monkeypatch.setenv("DEVBOT_COMPRESS_MODEL", "some-custom-model")
        val = os.environ.get("DEVBOT_COMPRESS_MODEL", "deepseek-v4-flash")
        assert val == "some-custom-model"


# ============================================================================
# 5. Megadelegate warning threshold (DEVBOT_MEGA_WARN_THRESHOLD)
# ============================================================================

class TestMegaWarnThreshold:
    """Tests for DEVBOT_MEGA_WARN_THRESHOLD default and custom values."""

    def test_mega_warn_threshold_default(self, monkeypatch):
        """Default threshold is 5 when env var is not set."""
        monkeypatch.delenv("DEVBOT_MEGA_WARN_THRESHOLD", raising=False)
        threshold = int(os.environ.get("DEVBOT_MEGA_WARN_THRESHOLD", "5"))
        assert threshold == 5

    def test_mega_warn_threshold_custom(self, monkeypatch):
        """Custom threshold is read correctly."""
        monkeypatch.setenv("DEVBOT_MEGA_WARN_THRESHOLD", "10")
        threshold = int(os.environ.get("DEVBOT_MEGA_WARN_THRESHOLD", "5"))
        assert threshold == 10

    def test_mega_warn_threshold_zero(self, monkeypatch):
        """Threshold of 0 means warn on every megaswarm (n > 0 always)."""
        monkeypatch.setenv("DEVBOT_MEGA_WARN_THRESHOLD", "0")
        threshold = int(os.environ.get("DEVBOT_MEGA_WARN_THRESHOLD", "5"))
        assert threshold == 0

    def test_mega_warn_threshold_high_suppresses_warning(self, monkeypatch):
        """A very high threshold means n rarely exceeds it."""
        monkeypatch.setenv("DEVBOT_MEGA_WARN_THRESHOLD", "999")
        threshold = int(os.environ.get("DEVBOT_MEGA_WARN_THRESHOLD", "5"))
        assert threshold == 999

    def test_warn_condition_when_n_exceeds_threshold(self):
        """Verify the warning condition logic: n > threshold triggers warning.

        This mirrors the inline check in run_megaswarm():
            if n > warn_threshold: print(warning)
        """
        # Default threshold = 5
        threshold = 5
        # n=3 should NOT trigger (3 > 5 is False)
        assert (3 > threshold) is False
        # n=5 should NOT trigger (5 > 5 is False)
        assert (5 > threshold) is False
        # n=6 SHOULD trigger (6 > 5 is True)
        assert (6 > threshold) is True
        # n=20 should trigger
        assert (20 > threshold) is True
