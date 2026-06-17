"""Tests for devbot.evolve_limits — the auto-evolve StopController."""

import os
import time

import pytest

from devbot.evolve_limits import StopController


# ============================================================================
# TestDefaultValues
# ============================================================================

class TestDefaultValues:
    """A fresh StopController has correct defaults."""

    def test_phase_count_zero(self):
        sc = StopController()
        assert sc.phase_count == 0

    def test_max_phases_default(self):
        sc = StopController()
        assert sc.max_phases == 20

    def test_time_limit_minutes_default(self):
        sc = StopController()
        assert sc.time_limit_minutes == 0

    def test_should_stop_returns_false_initially(self):
        sc = StopController()
        stopped, reason = sc.should_stop()
        assert stopped is False
        assert reason == ""


# ============================================================================
# TestMaxPhases
# ============================================================================

class TestMaxPhases:
    """When DEVBOT_EVOLVE_MAX_PHASES is set, phases are counted."""

    def test_below_max_no_stop(self, monkeypatch):
        monkeypatch.setenv("DEVBOT_EVOLVE_MAX_PHASES", "5")
        sc = StopController()
        for _ in range(3):
            sc.record_phase()
        stopped, _ = sc.should_stop()
        assert stopped is False

    def test_exactly_max_stops(self, monkeypatch):
        monkeypatch.setenv("DEVBOT_EVOLVE_MAX_PHASES", "5")
        sc = StopController()
        for _ in range(5):
            sc.record_phase()
        stopped, reason = sc.should_stop()
        assert stopped is True
        assert "Max phases reached" in reason
        assert "5/5" in reason

    def test_above_max_stops(self, monkeypatch):
        monkeypatch.setenv("DEVBOT_EVOLVE_MAX_PHASES", "3")
        sc = StopController()
        for _ in range(7):
            sc.record_phase()
        stopped, reason = sc.should_stop()
        assert stopped is True
        assert "Max phases reached" in reason

    def test_max_phases_zero_unlimited(self, monkeypatch):
        """max_phases=0 means no cap from phases."""
        monkeypatch.setenv("DEVBOT_EVOLVE_MAX_PHASES", "0")
        sc = StopController()
        for _ in range(100):
            sc.record_phase()
        stopped, _ = sc.should_stop()
        assert stopped is False  # 0 means unlimited


# ============================================================================
# TestTimeLimit
# ============================================================================

class TestTimeLimit:
    """When DEVBOT_EVOLVE_TIME_LIMIT is set, wall-clock is honoured."""

    def test_before_time_limit_no_stop(self, monkeypatch):
        monkeypatch.setenv("DEVBOT_EVOLVE_TIME_LIMIT", "10")
        # Pin the clock to a small, exact base before start() so elapsed
        # arithmetic is free of float cancellation (real monotonic() values
        # are large and (base + delta) - base can drift below delta).
        monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
        sc = StopController()
        sc.start()
        # Simulate only 1 minute elapsed (limit is 10 min)
        monkeypatch.setattr(time, "monotonic", lambda: 1000.0 + 60)
        stopped, _ = sc.should_stop()
        assert stopped is False

    def test_after_time_limit_stops(self, monkeypatch):
        monkeypatch.setenv("DEVBOT_EVOLVE_TIME_LIMIT", "2")
        monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
        sc = StopController()
        sc.start()
        # Simulate 2 minutes elapsed exactly (limit is 2 min = 120 s)
        monkeypatch.setattr(time, "monotonic", lambda: 1000.0 + 120)
        stopped, reason = sc.should_stop()
        assert stopped is True
        assert "Time limit exceeded" in reason
        assert "2" in reason

    def test_past_time_limit_stops(self, monkeypatch):
        monkeypatch.setenv("DEVBOT_EVOLVE_TIME_LIMIT", "1")
        monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
        sc = StopController()
        sc.start()
        # Way past the limit
        monkeypatch.setattr(time, "monotonic", lambda: 1000.0 + 999)
        stopped, reason = sc.should_stop()
        assert stopped is True
        assert "Time limit exceeded" in reason

    def test_time_limit_zero_unlimited(self, monkeypatch):
        """time_limit=0 means no time cap."""
        monkeypatch.setenv("DEVBOT_EVOLVE_TIME_LIMIT", "0")
        sc = StopController()
        sc.start()
        # Simulate a huge elapsed time
        fake_now = sc._start_time + 999_999_999
        monkeypatch.setattr(time, "monotonic", lambda: fake_now)
        stopped, _ = sc.should_stop()
        assert stopped is False  # 0 = unlimited

    def test_start_not_called_skips_time_check(self, monkeypatch):
        """Without start(), the time check is skipped entirely."""
        monkeypatch.setenv("DEVBOT_EVOLVE_TIME_LIMIT", "1")
        sc = StopController()
        # Never call start()
        # Even if we monkeypatch time, should not crash and should not stop
        monkeypatch.setattr(time, "monotonic", lambda: 999_999_999.0)
        stopped, _ = sc.should_stop()
        assert stopped is False  # start_time is None → skip


# ============================================================================
# TestGlobalBudget
# ============================================================================

class TestGlobalBudget:
    """The controller honours the global token budget."""

    def test_budget_not_exceeded_no_stop(self, monkeypatch):
        monkeypatch.setattr(
            "devbot.evolve_limits.check_global_budget_exceeded",
            lambda: False,
        )
        sc = StopController()
        stopped, _ = sc.should_stop()
        assert stopped is False

    def test_budget_exceeded_stops(self, monkeypatch):
        monkeypatch.setattr(
            "devbot.evolve_limits.check_global_budget_exceeded",
            lambda: True,
        )
        monkeypatch.setattr(
            "devbot.evolve_limits.get_global_token_count",
            lambda: 123_456,
        )
        monkeypatch.setenv("DEVBOT_GLOBAL_BUDGET", "100000")
        sc = StopController()
        stopped, reason = sc.should_stop()
        assert stopped is True
        assert "Global token budget exhausted" in reason
        assert "123,456" in reason
        assert "100,000" in reason


# ============================================================================
# TestStopOnRed
# ============================================================================

class TestStopOnRed:
    """The stop-on-red flag triggers immediately."""

    def test_after_set_red_stops(self):
        sc = StopController()
        sc.set_red()
        stopped, reason = sc.should_stop()
        assert stopped is True
        assert "Stop-on-red" in reason

    def test_before_set_red_no_stop(self):
        sc = StopController()
        stopped, _ = sc.should_stop()
        assert stopped is False


# ============================================================================
# TestRecordPhase
# ============================================================================

class TestRecordPhase:
    """record_phase increments the counter."""

    def test_zero_calls_count_zero(self):
        sc = StopController()
        assert sc.phase_count == 0

    def test_five_calls_count_five(self):
        sc = StopController()
        for _ in range(5):
            sc.record_phase()
        assert sc.phase_count == 5


# ============================================================================
# TestEnvVarConfiguration
# ============================================================================

class TestEnvVarConfiguration:
    """Custom env vars are read correctly at __init__ time."""

    def test_max_phases_from_env(self, monkeypatch):
        monkeypatch.setenv("DEVBOT_EVOLVE_MAX_PHASES", "5")
        sc = StopController()
        assert sc.max_phases == 5

    def test_max_phases_zero_means_unlimited(self, monkeypatch):
        monkeypatch.setenv("DEVBOT_EVOLVE_MAX_PHASES", "0")
        sc = StopController()
        assert sc.max_phases == 0

    def test_time_limit_from_env(self, monkeypatch):
        monkeypatch.setenv("DEVBOT_EVOLVE_TIME_LIMIT", "30")
        sc = StopController()
        assert sc.time_limit_minutes == 30


# ============================================================================
# TestOrderOfChecks
# ============================================================================

class TestOrderOfChecks:
    """The first triggered condition wins in should_stop()."""

    def test_red_wins_over_max_phases(self, monkeypatch):
        """If both red and max_phases are hit, red reason is returned."""
        monkeypatch.setenv("DEVBOT_EVOLVE_MAX_PHASES", "1")
        monkeypatch.setattr(
            "devbot.evolve_limits.check_global_budget_exceeded",
            lambda: False,
        )
        sc = StopController()
        sc.record_phase()  # hits max_phases=1
        sc.set_red()
        stopped, reason = sc.should_stop()
        assert stopped is True
        assert "Stop-on-red" in reason  # checked first

    def test_budget_wins_over_max_phases(self, monkeypatch):
        """Budget is checked before max_phases."""
        monkeypatch.setenv("DEVBOT_EVOLVE_MAX_PHASES", "1")
        monkeypatch.setattr(
            "devbot.evolve_limits.check_global_budget_exceeded",
            lambda: True,
        )
        monkeypatch.setattr(
            "devbot.evolve_limits.get_global_token_count",
            lambda: 100,
        )
        monkeypatch.setenv("DEVBOT_GLOBAL_BUDGET", "100")
        sc = StopController()
        sc.record_phase()  # would also trigger max_phases
        stopped, reason = sc.should_stop()
        assert stopped is True
        assert "Global token budget" in reason  # checked second, before max_phases

    def test_max_phases_wins_over_time(self, monkeypatch):
        """Max phases is checked before time limit."""
        monkeypatch.setenv("DEVBOT_EVOLVE_MAX_PHASES", "1")
        monkeypatch.setenv("DEVBOT_EVOLVE_TIME_LIMIT", "1")
        monkeypatch.setattr(
            "devbot.evolve_limits.check_global_budget_exceeded",
            lambda: False,
        )
        sc = StopController()
        sc.start()
        sc.record_phase()  # triggers max_phases
        # Also make time elapsed past limit
        fake_now = sc._start_time + 999
        monkeypatch.setattr(time, "monotonic", lambda: fake_now)
        stopped, reason = sc.should_stop()
        assert stopped is True
        assert "Max phases reached" in reason  # checked third, before time


# ============================================================================
# TestStartMethod
# ============================================================================

class TestStartMethod:
    """start() sets the start time so the time check becomes active."""

    def test_before_start_no_crash_on_should_stop(self):
        """should_stop() with a time limit but no start() doesn't crash."""
        sc = StopController()
        sc._time_limit_minutes = 1
        sc._time_limit_seconds = 60
        # _start_time is None → time check skipped
        stopped, _ = sc.should_stop()
        assert stopped is False

    def test_after_start_time_check_becomes_active(self, monkeypatch):
        """After start(), the time check actually fires."""
        monkeypatch.setenv("DEVBOT_EVOLVE_TIME_LIMIT", "1")
        sc = StopController()
        sc.start()
        # simulate 2 minutes elapsed
        fake_now = sc._start_time + 120
        monkeypatch.setattr(time, "monotonic", lambda: fake_now)
        stopped, reason = sc.should_stop()
        assert stopped is True
        assert "Time limit exceeded" in reason
