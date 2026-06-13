"""CLI entry-point tests."""

from __future__ import annotations

import sys

import pytest

import devbot.cli as cli


def test_auto_evolve_flag_calls_run_evolve(monkeypatch, tmp_path):
    calls = []

    def fake_run_evolve(root, model):
        calls.append((root, model))
        return True

    monkeypatch.setattr("devbot.evolve.run_evolve", fake_run_evolve)
    monkeypatch.setattr(
        sys,
        "argv",
        ["devbot", "--auto-evolve", "-C", str(tmp_path), "-m", "deepseek-v4-pro"],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    assert calls == [(tmp_path.resolve(), "deepseek-v4-pro")]


def test_auto_evolve_flag_exits_nonzero_on_failure(monkeypatch, tmp_path):
    monkeypatch.setattr("devbot.evolve.run_evolve", lambda root, model: False)
    monkeypatch.setattr(sys, "argv", ["devbot", "--auto-evolve", "-C", str(tmp_path)])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
