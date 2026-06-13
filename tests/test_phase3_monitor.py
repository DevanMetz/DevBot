"""Smoke tests for the ParallelMonitor dashboard (Phase 3)."""
import sys
from devbot.swarm import ParallelMonitor, AgentStatus

# Stash the real isatty so we can restore it.
_original_isatty = sys.stdout.isatty


def _force_tty():
    sys.stdout.isatty = lambda: True


def _restore_tty():
    sys.stdout.isatty = _original_isatty


def test_basic_line_building():
    _force_tty()
    try:
        labels = ['coder', 'researcher', 'tester']
        monitor = ParallelMonitor(labels)

        assert monitor._n == 3
        assert monitor._is_tty is True
        assert monitor._collapse is False

        monitor.update('coder', phase='running',
                       current_tool='\u23ba read_file plan.md')
        monitor.update('researcher', phase='running',
                       last_snippet='Looking at swarm.py...')
        monitor.update('tester', phase='done', tokens=1234)

        lines = monitor._build_lines()
        plain = [l.replace('\x1b', '') for l in lines]  # strip ANSI for assertions
        assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}: {lines}"
        assert 'coder' in plain[0]
        assert 'researcher' in plain[1]
        assert 'tester' in plain[2]
        assert '1,234t' in plain[2], f"Token count missing: {plain[2]}"

        print("  [OK] line building")
    finally:
        _restore_tty()


def test_hook_closures():
    _force_tty()
    try:
        monitor = ParallelMonitor(['coder'])
        on_text, on_tool_start, on_tool_end = monitor.get_hooks('coder')

        # Before first update, statuses may be empty; hooks handle gracefully
        on_text('hello world')
        monitor.update('coder', phase='running')
        on_text('hello world')
        st = monitor.statuses['coder']
        assert 'hello world' in st.last_snippet

        on_tool_start('edit_file', {'path': 'foo.py'})
        assert 'edit_file' in (st.current_tool or '')

        on_tool_end('done')
        assert st.current_tool is None

        print("  [OK] hook closures")
    finally:
        _restore_tty()


def test_collapse_mode():
    _force_tty()
    try:
        nine = [f'a{i}' for i in range(1, 10)]
        monitor = ParallelMonitor(nine)
        assert monitor._collapse is True, "9 > 8 should trigger collapse"

        for i, lbl in enumerate(nine):
            monitor.update(lbl, phase='running' if i < 6 else 'done')

        lines = monitor._build_lines()
        # First line = summary, up to 4 detail lines = 5 max
        assert 2 <= len(lines) <= monitor._MAX_COLLAPSED_LINES
        assert '6 running' in lines[0], f"Summary: {lines[0]}"
        assert '3 done' in lines[0], f"Summary: {lines[0]}"

        print("  [OK] collapse mode")
    finally:
        _restore_tty()


def test_non_tty_fallback():
    # Don't force TTY — ensure the monitor detects non-TTY and stays quiet
    assert _original_isatty is not None
    monitor = ParallelMonitor(['coder'])
    if not sys.stdout.isatty():
        # On a real TTY this won't hold, but the start/stop should be no-ops
        assert monitor._is_tty is False
    print("  [OK] non-TTY path")


if __name__ == '__main__':
    print("Phase 3 ParallelMonitor unit tests:")
    test_basic_line_building()
    test_hook_closures()
    test_collapse_mode()
    test_non_tty_fallback()
    print("All passed.")
