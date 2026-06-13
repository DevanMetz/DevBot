"""Unit tests for session persistence (Phase 3). No network calls."""

import json
import os
from pathlib import Path

import pytest

from devbot.session import (
    _sessions_dir,
    _ensure_dir,
    _make_session_id,
    _session_path,
    save_session,
    load_session,
    list_sessions,
    get_latest_session_id,
    restore_agent,
    SESSION_PREFIX,
    SESSIONS_DIR,
)


class TestSessionIdAndPaths:
    def test_make_session_id_format(self):
        sid = _make_session_id()
        assert sid.startswith(SESSION_PREFIX)
        # Should be like: session-20250115-143022-0000ab12
        # (date, time, then a counter+random uniqueness suffix)
        parts = sid[len(SESSION_PREFIX):].split("-")
        assert len(parts) == 3  # date, time, uniqueness suffix
        assert len(parts[0]) == 8 and parts[0].isdigit()  # YYYYMMDD
        assert len(parts[1]) == 6 and parts[1].isdigit()  # HHMMSS
        assert parts[2] and parts[2].isalnum()  # counter + random hex

    def test_sessions_dir(self, tmp_path):
        assert _sessions_dir(tmp_path) == tmp_path / SESSIONS_DIR

    def test_ensure_dir_creates(self, tmp_path):
        d = _ensure_dir(tmp_path)
        assert d.is_dir()
        assert d.name == SESSIONS_DIR

    def test_session_path(self, tmp_path):
        p = _session_path(tmp_path, "session-20250115-120000-000001")
        assert p == tmp_path / SESSIONS_DIR / "session-20250115-120000-000001.json"

    def test_make_session_id_unique(self):
        ids = {_make_session_id() for _ in range(100)}
        assert len(ids) == 100  # all unique


class TestSaveAndLoad:
    def test_save_and_load_roundtrip(self, tmp_path):
        """Simulate saving and loading a session via raw JSON (no Agent needed)."""
        d = _ensure_dir(tmp_path)
        sid = "session-test"
        data = {
            "id": sid,
            "created": "2025-01-15T12:00:00",
            "updated": "2025-01-15T12:05:00",
            "model": "deepseek-v4-flash",
            "swarm": False,
            "megaswarm": False,
            "auto_approve": True,
            "total_tokens": 5000,
            "last_prompt_tokens": 200,
            "delegation_count": 3,
            "messages": [
                {"role": "system", "content": "You are DevBot."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        }
        path = d / f"{sid}.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        loaded = load_session(tmp_path, sid)
        assert loaded is not None
        assert loaded["id"] == sid
        assert loaded["model"] == "deepseek-v4-flash"
        assert loaded["total_tokens"] == 5000
        assert len(loaded["messages"]) == 3

    def test_load_nonexistent(self, tmp_path):
        assert load_session(tmp_path, "nonexistent") is None

    def test_load_corrupted(self, tmp_path):
        d = _ensure_dir(tmp_path)
        (d / "session-bad.json").write_text("not json", encoding="utf-8")
        assert load_session(tmp_path, "session-bad") is None


class TestListSessions:
    def test_empty_dir(self, tmp_path):
        assert list_sessions(tmp_path) == []

    def test_dir_does_not_exist(self, tmp_path):
        assert list_sessions(tmp_path / "nonexistent") == []

    def test_list_multiple(self, tmp_path):
        d = _ensure_dir(tmp_path)
        # Create sessions with different timestamps
        sessions = [
            ("session-20250115-120000", {
                "id": "session-20250115-120000",
                "created": "2025-01-15T12:00:00",
                "updated": "2025-01-15T12:05:00",
                "model": "deepseek-v4-flash",
                "swarm": False,
                "megaswarm": False,
                "total_tokens": 1000,
                "messages": [{"role": "user", "content": "a"}] * 5,
            }),
            ("session-20250115-130000", {
                "id": "session-20250115-130000",
                "created": "2025-01-15T13:00:00",
                "updated": "2025-01-15T13:10:00",
                "model": "deepseek-v4-pro",
                "swarm": True,
                "megaswarm": False,
                "total_tokens": 50000,
                "messages": [{"role": "user", "content": "b"}] * 20,
            }),
            ("session-20250114-090000", {
                "id": "session-20250114-090000",
                "created": "2025-01-14T09:00:00",
                "updated": "2025-01-14T09:30:00",
                "model": "deepseek-v4-flash",
                "swarm": False,
                "megaswarm": True,
                "total_tokens": 100000,
                "messages": [{"role": "user", "content": "c"}] * 100,
            }),
        ]
        for sid, data in sessions:
            (d / f"{sid}.json").write_text(json.dumps(data), encoding="utf-8")

        result = list_sessions(tmp_path)
        assert len(result) == 3

        # Should be newest first (sorted by filename descending)
        assert result[0]["id"] == "session-20250115-130000"
        assert result[1]["id"] == "session-20250115-120000"
        assert result[2]["id"] == "session-20250114-090000"

        # Check metadata
        assert result[0]["model"] == "deepseek-v4-pro"
        assert result[0]["swarm"] is True
        assert result[0]["message_count"] == 20
        assert result[0]["total_tokens"] == 50000

        assert result[2]["megaswarm"] is True
        assert result[2]["message_count"] == 100
        assert result[2]["total_tokens"] == 100000

    def test_skips_corrupted(self, tmp_path):
        d = _ensure_dir(tmp_path)
        (d / "session-good.json").write_text(
            json.dumps({"id": "session-good", "model": "x", "messages": []}),
            encoding="utf-8")
        (d / "session-bad.json").write_text("garbage", encoding="utf-8")

        result = list_sessions(tmp_path)
        assert len(result) == 1
        assert result[0]["id"] == "session-good"

    def test_ignores_non_session_files(self, tmp_path):
        d = _ensure_dir(tmp_path)
        (d / "other.txt").write_text("hello", encoding="utf-8")
        (d / "session-x.json").write_text(
            json.dumps({"id": "session-x", "model": "m", "messages": []}),
            encoding="utf-8")

        result = list_sessions(tmp_path)
        assert len(result) == 1
        assert result[0]["id"] == "session-x"


class TestGetLatest:
    def test_none_when_empty(self, tmp_path):
        assert get_latest_session_id(tmp_path) is None

    def test_returns_newest(self, tmp_path):
        d = _ensure_dir(tmp_path)
        (d / "session-old.json").write_text(
            json.dumps({"id": "session-old", "model": "x", "created": "2025-01-15T10:00:00", "messages": []}),
            encoding="utf-8")
        (d / "session-new.json").write_text(
            json.dumps({"id": "session-new", "model": "y", "created": "2025-01-15T11:00:00", "messages": []}),
            encoding="utf-8")

        assert get_latest_session_id(tmp_path) == "session-new"


class TestSaveSessionWithAgent:
    """Tests that exercise save_session with a real Agent (no API calls)."""

    def test_save_skips_labeled_agent(self, tmp_path):
        """Sub-agents (with labels) should be skipped."""
        from devbot.agent import Agent

        # We can't easily create an Agent without an API key, so test the
        # guard directly: an agent with label set returns None.
        agent = Agent.__new__(Agent)
        agent.label = "coder"
        agent.root = tmp_path
        agent.token_budget = 0
        result = save_session(agent)
        assert result is None

    def test_save_creates_file(self, tmp_path):
        """A main agent gets persisted to .devbot/."""
        from devbot.agent import Agent

        agent = Agent.__new__(Agent)
        agent.label = None
        agent.root = tmp_path
        agent.model = "deepseek-v4-flash"
        agent.swarm = False
        agent.megaswarm = True
        agent.auto_approve = False
        agent.total_tokens = 12345
        agent.last_prompt_tokens = 200
        agent.delegation_count = 5
        agent.token_budget = 0
        agent.messages = [
            {"role": "system", "content": "You are DevBot."},
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]
        agent.session_id = None  # will be auto-generated

        sid = save_session(agent)
        assert sid is not None
        assert sid.startswith(SESSION_PREFIX)
        assert agent.session_id == sid

        # Verify the file exists and contains the right data
        path = tmp_path / SESSIONS_DIR / f"{sid}.json"
        assert path.is_file()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["id"] == sid
        assert data["model"] == "deepseek-v4-flash"
        assert data["megaswarm"] is True
        assert data["total_tokens"] == 12345
        assert data["delegation_count"] == 5
        assert len(data["messages"]) == 3

    def test_save_reuses_session_id(self, tmp_path):
        """Second save reuses the existing session_id."""
        from devbot.agent import Agent

        agent = Agent.__new__(Agent)
        agent.label = None
        agent.root = tmp_path
        agent.model = "deepseek-v4-flash"
        agent.swarm = False
        agent.megaswarm = False
        agent.auto_approve = False
        agent.total_tokens = 100
        agent.last_prompt_tokens = 50
        agent.delegation_count = 0
        agent.token_budget = 0
        agent.messages = [{"role": "user", "content": "hi"}]
        agent.session_id = None

        sid1 = save_session(agent)
        assert sid1 is not None
        assert agent.session_id == sid1

        # Second save should reuse same id
        agent.total_tokens = 200
        sid2 = save_session(agent)
        assert sid2 == sid1

        # File should be updated
        path = tmp_path / SESSIONS_DIR / f"{sid1}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["total_tokens"] == 200


class TestRestoreAgent:
    """Test restore_agent creates a properly configured Agent from saved data."""

    def test_restore_from_saved_data(self, tmp_path):
        """Create a saved session, then restore it."""
        d = _ensure_dir(tmp_path)
        data = {
            "id": "session-restore-test",
            "created": "2025-01-15T12:00:00",
            "updated": "2025-01-15T12:30:00",
            "model": "deepseek-v4-pro",
            "swarm": True,
            "megaswarm": False,
            "auto_approve": True,
            "total_tokens": 99999,
            "last_prompt_tokens": 500,
            "delegation_count": 7,
            "messages": [
                {"role": "system", "content": "You are DevBot."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
        }
        (d / "session-restore-test.json").write_text(
            json.dumps(data), encoding="utf-8")

        agent = restore_agent(tmp_path, "session-restore-test")
        assert agent is not None
        assert agent.model == "deepseek-v4-pro"
        assert agent.swarm is True
        assert agent.megaswarm is False
        assert agent.total_tokens == 99999
        assert agent.last_prompt_tokens == 500
        assert agent.delegation_count == 7
        assert len(agent.messages) == 3
        assert agent.messages[1]["content"] == "Hello"
        assert agent.session_id == "session-restore-test"
        assert agent.session_created == "2025-01-15T12:00:00"

    def test_restore_nonexistent(self, tmp_path):
        assert restore_agent(tmp_path, "nonexistent") is None

    def test_restore_latest_none(self, tmp_path):
        assert restore_agent(tmp_path, session_id=None) is None

    def test_restore_latest(self, tmp_path):
        d = _ensure_dir(tmp_path)
        (d / "session-first.json").write_text(json.dumps({
            "id": "session-first",
            "model": "deepseek-v4-flash",
            "created": "2025-01-15T09:00:00",
            "messages": [],
        }), encoding="utf-8")
        (d / "session-second.json").write_text(json.dumps({
            "id": "session-second",
            "model": "deepseek-v4-pro",
            "created": "2025-01-15T10:00:00",
            "messages": [{"role": "user", "content": "latest"}],
        }), encoding="utf-8")

        agent = restore_agent(tmp_path, session_id=None)
        assert agent is not None
        assert agent.session_id == "session-second"
        assert agent.model == "deepseek-v4-pro"


class TestSessionIntegration:
    """End-to-end: Agent.run() saves, then restore picks it up."""

    def test_run_saves_and_restores(self, tmp_path, monkeypatch):
        """Simulate a full save/restore cycle.

        We mock the API to avoid network calls, then verify that after
        agent.run() the session file exists and can be restored.
        """
        # Set a fake API key so Agent.__init__ doesn't exit
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake-for-test")

        from unittest.mock import MagicMock, patch
        from devbot.agent import Agent

        # We need to mock the client's chat.completions.create to return
        # an empty stream (no tool calls) so run() finishes cleanly.
        with patch.object(Agent, '_create_stream', autospec=True) as mock_stream:
            # Build a mock that simulates a simple text response with no tools
            mock_chunk = MagicMock()
            mock_chunk.usage = None
            mock_chunk.choices = [
                MagicMock(delta=MagicMock(
                    reasoning_content=None,
                    content="Hello, world!",
                    tool_calls=None,
                ), finish_reason="stop")
            ]

            # Second chunk carries usage
            mock_usage_chunk = MagicMock()
            mock_usage_chunk.usage = MagicMock(prompt_tokens=10, total_tokens=20)
            mock_usage_chunk.choices = []

            mock_stream.return_value = iter([mock_chunk, mock_usage_chunk])

            agent = Agent(root=tmp_path, model="deepseek-v4-flash",
                          auto_approve=True)
            assert agent.session_id is None
            assert agent.label is None  # main agent

            result = agent.run("Say hello")
            assert result == "Hello, world!"

            # After run(), session should be saved
            assert agent.session_id is not None
            sid = agent.session_id
            path = tmp_path / SESSIONS_DIR / f"{sid}.json"
            assert path.is_file()

            # Verify the saved data
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["id"] == sid
            assert len(data["messages"]) >= 3  # system + user + assistant
            assert data["messages"][-1]["role"] == "assistant"

            # Now restore and check
            restored = restore_agent(tmp_path, sid)
            assert restored is not None
            assert restored.session_id == sid
            assert restored.total_tokens == agent.total_tokens
            assert len(restored.messages) == len(agent.messages)

    def test_sub_agent_not_saved(self, tmp_path, monkeypatch):
        """Sub-agents (with labels) should NOT trigger session saves."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake-for-test")

        from unittest.mock import MagicMock, patch
        from devbot.agent import Agent

        with patch.object(Agent, '_create_stream', autospec=True) as mock_stream:
            mock_chunk = MagicMock()
            mock_chunk.usage = None
            mock_chunk.choices = [
                MagicMock(delta=MagicMock(
                    reasoning_content=None,
                    content="Done.",
                    tool_calls=None,
                ), finish_reason="stop")
            ]
            mock_usage_chunk = MagicMock()
            mock_usage_chunk.usage = MagicMock(prompt_tokens=5, total_tokens=10)
            mock_usage_chunk.choices = []
            mock_stream.return_value = iter([mock_chunk, mock_usage_chunk])

            agent = Agent(root=tmp_path, model="deepseek-v4-flash",
                          auto_approve=True, label="coder")
            agent.run("Do something")

            # No session files should be created for a sub-agent
            sessions_dir = tmp_path / SESSIONS_DIR
            session_files = list(sessions_dir.glob(f"{SESSION_PREFIX}*.json")) if sessions_dir.is_dir() else []
            assert len(session_files) == 0


class TestAtomicWrite:
    """Verify the save uses atomic-write (tmp then rename) pattern."""

    def test_no_tmp_left_behind(self, tmp_path):
        from devbot.agent import Agent

        agent = Agent.__new__(Agent)
        agent.label = None
        agent.root = tmp_path
        agent.model = "m"
        agent.swarm = False
        agent.megaswarm = False
        agent.auto_approve = False
        agent.total_tokens = 0
        agent.last_prompt_tokens = 0
        agent.delegation_count = 0
        agent.token_budget = 0
        agent.messages = []
        agent.session_id = None

        save_session(agent)

        # No .tmp files should remain
        tmps = list((tmp_path / SESSIONS_DIR).glob("*.tmp"))
        assert len(tmps) == 0

        # The session file should exist
        sessions = list((tmp_path / SESSIONS_DIR).glob(f"{SESSION_PREFIX}*.json"))
        assert len(sessions) == 1
