"""CLI entry-point tests."""

from __future__ import annotations

import sys

import pytest

import devbot.cli as cli


def test_auto_evolve_flag_calls_run_evolve(monkeypatch, tmp_path):
    calls = []

    def fake_run_evolve(root, model, provider=None):
        calls.append((root, model, provider))
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
    assert calls == [(tmp_path.resolve(), "deepseek-v4-pro", None)]


def test_auto_evolve_flag_exits_nonzero_on_failure(monkeypatch, tmp_path):
    monkeypatch.setattr("devbot.evolve.run_evolve", lambda root, model, provider=None: False)
    monkeypatch.setattr(sys, "argv", ["devbot", "--auto-evolve", "-C", str(tmp_path)])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1


def test_local_flag_passes_provider_to_auto_evolve(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "devbot.evolve.run_evolve",
        lambda root, model, provider=None: calls.append((root, model, provider)) or True,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["devbot", "--local", "--auto-evolve", "-C", str(tmp_path)],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    assert calls == [(tmp_path.resolve(), None, "local-vibethinker")]


def test_local_flag_uses_local_provider(monkeypatch, tmp_path):
    calls = []

    class FakeAgent:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.model = "vibethinker-q4-vulkan"
            self.provider_display_name = "VibeThinker Local"
            self.megaswarm = False
            self.swarm = False

        def run(self, prompt):
            calls.append({"prompt": prompt})

    monkeypatch.setattr(cli, "Agent", FakeAgent)
    monkeypatch.setattr(sys, "argv", ["devbot", "--local", "-C", str(tmp_path), "hello"])

    cli.main()

    assert calls[0]["provider"] == "local-vibethinker"
    assert calls[0]["root"] == tmp_path.resolve()
    assert calls[1] == {"prompt": "hello"}


def test_think_toggles_for_local_provider(monkeypatch, tmp_path, capsys):
    agents = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.model = "vibethinker-q4-vulkan"
            self.provider_name = "local-vibethinker"
            self.provider_display_name = "VibeThinker Local"
            self.show_reasoning = False
            self.auto_approve = False
            self.megaswarm = False
            self.swarm = False
            agents.append(self)

    inputs = iter(["/think", "/exit"])
    monkeypatch.setattr(cli, "Agent", FakeAgent)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr(sys, "argv", ["devbot", "--local", "-C", str(tmp_path)])

    cli.main()

    out = capsys.readouterr().out
    assert "show reasoning: on | local VibeThinker may use more tokens" in out
    assert agents[0].show_reasoning is True
