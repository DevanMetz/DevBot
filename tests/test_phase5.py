"""Phase 5: Accurate cost estimation with cache-hit token accounting."""

import json
import os
import re
from pathlib import Path

import pytest

from devbot.agent import Agent, MODEL_PRICING


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================================
# Existing README parity tests (preserved from prior phase)
# ============================================================================

def test_readme_exists_and_contains_safety_model():
    """The README file should exist and document the safety model."""
    readme = PROJECT_ROOT / "README.md"
    assert readme.is_file(), "README.md not found"
    content = readme.read_text(encoding="utf-8")
    assert "Safety model" in content, "README.md missing 'Safety model' section"


# ---------------------------------------------------------------------------
# Helpers: extract DEVBOT_* env-var names from the README config table
# ---------------------------------------------------------------------------

def _readme_table_vars(readme_path: Path) -> set[str]:
    """Parse the README Configuration table and return all DEVBOT_* var names."""
    text = readme_path.read_text(encoding="utf-8")
    # Find the Configuration section
    m = re.search(r'## Configuration\n\n\| .*?\n\n', text, re.DOTALL)
    assert m, "Could not find '## Configuration' section with a table in README"
    table_text = m.group(0)
    # Each row starts with | `VAR_NAME` | ...
    vars_found = set()
    for line in table_text.splitlines():
        match = re.match(r'^\| `([^`]+)`', line.strip())
        if match:
            name = match.group(1)
            if name.startswith("DEVBOT_") or name == "DEEPSEEK_API_KEY":
                vars_found.add(name)
    return vars_found


# ---------------------------------------------------------------------------
# Helpers: extract DEVBOT_* env-var names from Python source files
# ---------------------------------------------------------------------------

def _source_vars(file_paths: list[Path]) -> set[str]:
    """Scan Python files for os.environ[...] / os.environ.get(...) of DEVBOT_* vars."""
    names: set[str] = set()
    # Matches: os.environ.get("DEVBOT_X", ...)  or  os.environ["DEVBOT_X"]
    # Also int(os.environ.get(...))
    pattern = re.compile(
        r'os\.environ(?:\.get)?\(\s*["\'](DEVBOT_[A-Z_]+)["\']'
    )
    for fp in file_paths:
        if not fp.is_file():
            continue
        content = fp.read_text(encoding="utf-8")
        for m in pattern.finditer(content):
            names.add(m.group(1))
    return names


# ---------------------------------------------------------------------------
# Actual README parity tests
# ---------------------------------------------------------------------------

def test_every_readme_env_var_exists_in_code():
    """No invented env vars in README — every listed DEVBOT_* must be in the code."""
    readme = PROJECT_ROOT / "README.md"
    readme_vars = _readme_table_vars(readme)

    src_files = [
        PROJECT_ROOT / "devbot" / "agent.py",
        PROJECT_ROOT / "devbot" / "swarm.py",
        PROJECT_ROOT / "devbot" / "devlog.py",
    ]
    code_vars = _source_vars(src_files)

    # DEEPSEEK_API_KEY is not a DEVBOT_* var — exclude it for the exact match.
    readme_devbot = {v for v in readme_vars if v.startswith("DEVBOT_")}

    # Every DEVBOT_* in the README must be found in the code.
    missing_in_code = readme_devbot - code_vars
    assert not missing_in_code, (
        f"README lists env vars not found in code: {sorted(missing_in_code)}"
    )


def test_every_code_env_var_exists_in_readme():
    """No undocumented env vars — every DEVBOT_* in code must be in the README table."""
    readme = PROJECT_ROOT / "README.md"
    readme_vars = _readme_table_vars(readme)

    src_files = [
        PROJECT_ROOT / "devbot" / "agent.py",
        PROJECT_ROOT / "devbot" / "swarm.py",
        PROJECT_ROOT / "devbot" / "devlog.py",
    ]
    code_vars = _source_vars(src_files)

    # Every DEVBOT_* in the code must be in the README.
    missing_in_readme = code_vars - readme_vars
    assert not missing_in_readme, (
        f"Code uses env vars not documented in README: {sorted(missing_in_readme)}"
    )


# ============================================================================
# New Phase 5 tests: cache-hit cost estimation
# ============================================================================

class TestCacheHitCostEstimation:
    """Test estimated_cost() with and without cache-hit breakdown."""

    def test_cost_with_cache_breakdown(self, tmp_path, monkeypatch):
        """With full breakdown, estimated_cost uses cache-hit pricing."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test1234567890abcdef")
        agent = Agent(root=tmp_path)
        agent.model = "deepseek-v4-flash"

        # Simulate a session with cache breakdown
        agent.total_tokens = 1_000_000
        agent.completion_tokens = 300_000
        agent.prompt_cache_hit_tokens = 500_000
        agent.prompt_cache_miss_tokens = 200_000
        # prompt_tokens = 500_000 + 200_000 = 700_000

        # Expected cost:
        # output: 300_000 * 0.28/1M = 0.084
        # cache_hit: 500_000 * 0.014/1M = 0.007
        # cache_miss: 200_000 * 0.14/1M = 0.028
        # total = 0.084 + 0.007 + 0.028 = 0.119
        cost = agent.estimated_cost()
        expected = (300_000 * 0.28 + 500_000 * 0.014 + 200_000 * 0.14) / 1_000_000
        assert cost == pytest.approx(expected)
        assert cost == pytest.approx(0.119)

    def test_cost_fallback_no_breakdown(self, tmp_path, monkeypatch):
        """Without breakdown counters, falls back to 50/50 split on total_tokens."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test1234567890abcdef")
        agent = Agent(root=tmp_path)
        agent.model = "deepseek-v4-flash"
        agent.total_tokens = 1_000_000
        # completion_tokens, prompt_cache_hit_tokens, prompt_cache_miss_tokens all 0

        cost = agent.estimated_cost()
        # Fallback: half * input + half * output
        expected = (500_000 * 0.14 + 500_000 * 0.28) / 1_000_000
        assert cost == pytest.approx(expected)
        assert cost == pytest.approx(0.21)

    def test_cost_with_pro_model(self, tmp_path, monkeypatch):
        """Cost calculation uses the correct model pricing."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test1234567890abcdef")
        agent = Agent(root=tmp_path, model="deepseek-v4-pro")
        agent.total_tokens = 1_000_000
        agent.completion_tokens = 400_000
        agent.prompt_cache_hit_tokens = 300_000
        agent.prompt_cache_miss_tokens = 300_000

        cost = agent.estimated_cost()
        expected = (
            400_000 * 0.87 +
            300_000 * 0.0435 +
            300_000 * 0.435
        ) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_cost_with_unknown_model_defaults_to_flash(self, tmp_path, monkeypatch):
        """Unknown model falls back to deepseek-v4-flash pricing."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test1234567890abcdef")
        agent = Agent(root=tmp_path, model="some-future-model")
        agent.total_tokens = 1_000_000
        agent.completion_tokens = 400_000
        agent.prompt_cache_hit_tokens = 300_000
        agent.prompt_cache_miss_tokens = 300_000

        cost = agent.estimated_cost()
        expected = (
            400_000 * 0.28 +
            300_000 * 0.014 +
            300_000 * 0.14
        ) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_cost_zero_tokens(self, tmp_path, monkeypatch):
        """Zero tokens gives zero cost."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test1234567890abcdef")
        agent = Agent(root=tmp_path)
        cost = agent.estimated_cost()
        assert cost == 0.0

    def test_cost_partial_breakdown_still_uses_breakdown(self, tmp_path, monkeypatch):
        """If any breakdown counter is > 0, the breakdown path is used
        (even if some counters are 0 — e.g. no cache hits)."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test1234567890abcdef")
        monkeypatch.setenv("DEVBOT_MODEL", "deepseek-v4-flash")
        agent = Agent(root=tmp_path)
        agent.model = "deepseek-v4-flash"
        agent.total_tokens = 700_000
        agent.completion_tokens = 300_000
        agent.prompt_cache_hit_tokens = 0
        agent.prompt_cache_miss_tokens = 400_000
        # prompt_tokens = 0 + 400_000 = 400_000

        cost = agent.estimated_cost()
        expected = (
            300_000 * 0.28 +
            0 * 0.014 +
            400_000 * 0.14
        ) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_model_pricing_has_cache_hit_keys(self):
        """Every model in MODEL_PRICING must have cache_hit_input key."""
        for model, pricing in MODEL_PRICING.items():
            assert "cache_hit_input" in pricing, (
                f"MODEL_PRICING['{model}'] missing 'cache_hit_input' key"
            )
            assert pricing["cache_hit_input"] > 0, (
                f"MODEL_PRICING['{model}']['cache_hit_input'] must be > 0"
            )
            assert pricing["cache_hit_input"] < pricing["input"], (
                f"MODEL_PRICING['{model}'] cache_hit_input must be < input price"
            )


class TestCacheHitPersistence:
    """Test that cache-hit counters survive save/restore round-trips."""

    def test_new_counters_saved_and_restored(self, tmp_path, monkeypatch):
        """save_session includes the new counters; restore_agent reads them back."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test1234567890abcdef")

        from devbot.session import save_session, restore_agent

        agent = Agent(root=tmp_path)
        agent.completion_tokens = 1234
        agent.prompt_cache_hit_tokens = 567
        agent.prompt_cache_miss_tokens = 890
        agent.total_tokens = 1234 + 567 + 890

        sid = save_session(agent)
        assert sid is not None

        restored = restore_agent(tmp_path, sid)
        assert restored is not None
        assert restored.completion_tokens == 1234
        assert restored.prompt_cache_hit_tokens == 567
        assert restored.prompt_cache_miss_tokens == 890
        assert restored.total_tokens == 1234 + 567 + 890

    def test_old_session_without_new_counters_defaults_to_zero(self, tmp_path, monkeypatch):
        """Loading a session saved before Phase 5 (without new counters)
        should default the new counters to 0."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test1234567890abcdef")
        monkeypatch.setenv("DEVBOT_SHOW_REASONING", "0")

        from devbot.session import save_session, restore_agent

        agent = Agent(root=tmp_path)
        # Don't set the new counters — simulate pre-Phase-5 session
        agent.total_tokens = 5000
        sid = save_session(agent)

        # Manually remove new counters from the saved JSON to simulate old format
        session_path = tmp_path / ".devbot" / f"{sid}.json"
        data = json.loads(session_path.read_text())
        data.pop("completion_tokens", None)
        data.pop("prompt_cache_hit_tokens", None)
        data.pop("prompt_cache_miss_tokens", None)
        session_path.write_text(json.dumps(data))

        restored = restore_agent(tmp_path, sid)
        assert restored is not None
        assert restored.completion_tokens == 0
        assert restored.prompt_cache_hit_tokens == 0
        assert restored.prompt_cache_miss_tokens == 0
        assert restored.total_tokens == 5000
