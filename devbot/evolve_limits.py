"""Hard-stop controller for the auto-evolve loop.

Centralises all stop conditions so the autopilot can check them in one
place each iteration.
"""

import os
import time

from devbot.agent import check_global_budget_exceeded, get_global_token_count


class StopController:
    """Tracks all stop conditions for the auto-evolve loop."""

    def __init__(self) -> None:
        self._max_phases = int(os.environ.get("DEVBOT_EVOLVE_MAX_PHASES", "20"))
        self._time_limit_minutes = int(os.environ.get("DEVBOT_EVOLVE_TIME_LIMIT", "0"))
        self._time_limit_seconds = self._time_limit_minutes * 60
        self._start_time: float | None = None
        self._phase_count = 0
        self._red = False  # stop-on-red flag

    # -- control methods --------------------------------------------------------

    def start(self) -> None:
        """Record the start time for the wall-clock deadline."""
        self._start_time = time.monotonic()

    def record_phase(self) -> None:
        """Increment the phase counter."""
        self._phase_count += 1

    def set_red(self) -> None:
        """Set the stop-on-red flag (e.g. from a failed pipeline phase)."""
        self._red = True

    # -- properties -------------------------------------------------------------

    @property
    def phase_count(self) -> int:
        return self._phase_count

    @property
    def max_phases(self) -> int:
        return self._max_phases

    @property
    def time_limit_minutes(self) -> int:
        return self._time_limit_minutes

    # -- stop logic -------------------------------------------------------------

    def should_stop(self) -> tuple[bool, str]:
        """Return (True, reason) if any stop condition is met.

        Checks in order:

        1. Stop-on-red (``set_red`` was called)
        2. Global token budget exhausted (via ``agent.check_global_budget_exceeded``)
        3. Max phases reached
        4. Time limit exceeded
        """
        # 1. Stop-on-red
        if self._red:
            return True, "Stop-on-red: a phase failed or was rejected"

        # 2. Global token budget
        if check_global_budget_exceeded():
            budget = int(os.environ.get("DEVBOT_GLOBAL_BUDGET", "0"))
            count = get_global_token_count()
            return True, f"Global token budget exhausted ({count:,} >= {budget:,})"

        # 3. Max phases
        if self._max_phases > 0 and self._phase_count >= self._max_phases:
            return True, f"Max phases reached ({self._phase_count}/{self._max_phases})"

        # 4. Time limit
        if self._start_time is not None and self._time_limit_seconds > 0:
            elapsed = time.monotonic() - self._start_time
            if elapsed >= self._time_limit_seconds:
                mins = self._time_limit_minutes
                return True, f"Time limit exceeded ({mins} minute{'s' if mins != 1 else ''})"

        return False, ""
