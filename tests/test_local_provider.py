import os

import pytest

from devbot.agent import (
    Agent,
    DEEPSEEK_BASE_URL,
    LOCAL_LLM_DEFAULT_MAX_TOKENS,
    LOCAL_LLM_DEFAULT_MODEL,
    LOCAL_PROVIDER_NAME,
    _clean_local_model_text,
    _is_trivial_greeting,
    _local_user_content,
    _show_local_model_text,
    get_llm_provider_settings,
)


_LOCAL_ENV_NAMES = (
    "DEVBOT_PROVIDER",
    "DEVBOT_MODEL",
    "LOCAL_LLM_BASE_URL",
    "LOCAL_LLM_MAX_TOKENS",
    "LOCAL_LLM_MODEL",
    "LOCAL_LLM_API_KEY",
)


@pytest.fixture(autouse=True)
def _restore_local_env():
    original = {name: os.environ.get(name) for name in _LOCAL_ENV_NAMES}
    yield
    for name, value in original.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _clear_local_env(monkeypatch):
    for name in _LOCAL_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_deepseek_remains_default_provider(tmp_path, monkeypatch):
    _clear_local_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    agent = Agent(root=tmp_path)

    assert agent.provider_name == "deepseek"
    assert agent.base_url == DEEPSEEK_BASE_URL
    assert agent.model == "deepseek-v4-flash"


def test_local_provider_uses_local_env_without_deepseek_key(tmp_path, monkeypatch):
    _clear_local_env(monkeypatch)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEVBOT_PROVIDER", "local-vibethinker")
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8092/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "vibethinker-q4-vulkan")
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "local")

    agent = Agent(root=tmp_path)

    assert agent.provider_name == LOCAL_PROVIDER_NAME
    assert agent.provider_display_name == "VibeThinker Local"
    assert agent.base_url == "http://127.0.0.1:8092/v1"
    assert agent.model == "vibethinker-q4-vulkan"
    assert agent.estimated_cost() == 0.0


def test_local_model_alias_selects_configured_local_model(monkeypatch):
    _clear_local_env(monkeypatch)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEVBOT_MODEL", "local-vibethinker")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "custom-local-model")

    settings = get_llm_provider_settings()

    assert settings.provider == LOCAL_PROVIDER_NAME
    assert settings.model == "custom-local-model"


def test_local_provider_defaults_match_vibethinker_server(tmp_path, monkeypatch):
    _clear_local_env(monkeypatch)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEVBOT_PROVIDER", "local-vibethinker")

    agent = Agent(root=tmp_path)

    assert agent.base_url == "http://127.0.0.1:8092/v1"
    assert agent.model == LOCAL_LLM_DEFAULT_MODEL


def test_local_provider_can_be_selected_without_env(tmp_path, monkeypatch):
    _clear_local_env(monkeypatch)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    agent = Agent(root=tmp_path, provider="local-vibethinker")

    assert agent.provider_name == LOCAL_PROVIDER_NAME
    assert agent.base_url == "http://127.0.0.1:8092/v1"
    assert agent.model == LOCAL_LLM_DEFAULT_MODEL


def test_config_toml_can_select_local_provider(tmp_path, monkeypatch):
    _clear_local_env(monkeypatch)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    devbot_dir = tmp_path / ".devbot"
    devbot_dir.mkdir()
    devbot_dir.joinpath("config.toml").write_text(
        'provider = "local-vibethinker"\n'
        'local_llm_base_url = "http://127.0.0.1:8092/v1"\n'
        'local_llm_model = "vibethinker-q4-vulkan"\n'
        'local_llm_api_key = "local"\n',
        encoding="utf-8",
    )

    agent = Agent(root=tmp_path)

    assert agent.provider_name == LOCAL_PROVIDER_NAME
    assert os.environ["LOCAL_LLM_BASE_URL"] == "http://127.0.0.1:8092/v1"
    assert agent.model == "vibethinker-q4-vulkan"


def test_unknown_provider_is_rejected(monkeypatch):
    _clear_local_env(monkeypatch)
    monkeypatch.setenv("DEVBOT_PROVIDER", "bogus")

    with pytest.raises(SystemExit):
        get_llm_provider_settings()


def test_clean_local_model_text_strips_think_blocks():
    text = "<think>private reasoning</think>Hello there."

    cleaned, saw_text_tool = _clean_local_model_text(text)

    assert cleaned == "Hello there."
    assert saw_text_tool is False


def test_clean_local_model_text_suppresses_text_tool_wrappers():
    text = (
        "<think>private reasoning</think>"
        '<result><json>{ "name": "run_command", "arguments": { "command": "echo hi" }'
        "</json></result>"
    )

    cleaned, saw_text_tool = _clean_local_model_text(text)

    assert cleaned == ""
    assert saw_text_tool is True


def test_show_local_model_text_keeps_thinking():
    text = "<think>private reasoning</think>Visible answer."

    shown, saw_text_tool = _show_local_model_text(text)

    assert "[thinking]" in shown
    assert "private reasoning" in shown
    assert "Visible answer." in shown
    assert saw_text_tool is False


def test_trivial_greeting_bypasses_local_model(tmp_path, monkeypatch):
    _clear_local_env(monkeypatch)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    agent = Agent(root=tmp_path, provider="local-vibethinker")

    def fail_stream():
        raise AssertionError("trivial greeting should not call the model")

    monkeypatch.setattr(agent, "_stream_once", fail_stream)
    seen = []
    monkeypatch.setattr(agent, "on_text", seen.append)

    result = agent.run("hello")

    assert result == "Hello! What would you like to work on?"
    assert seen == ["Hello! What would you like to work on?\n"]


def test_local_stream_uses_default_max_tokens(tmp_path, monkeypatch):
    _clear_local_env(monkeypatch)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    agent = Agent(root=tmp_path, provider="local-vibethinker")
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return iter(())

    agent.client.chat.completions.create = fake_create

    agent._create_stream()

    assert captured["max_tokens"] == LOCAL_LLM_DEFAULT_MAX_TOKENS


def test_is_trivial_greeting():
    assert _is_trivial_greeting("hello")
    assert _is_trivial_greeting("hey!")
    assert not _is_trivial_greeting("hello, fix the tests")


def test_local_user_content_forces_no_think():
    content = _local_user_content("what do you think of devbot")

    assert content.startswith("/no_think")
    assert "Answer directly and briefly" in content
    assert "what do you think of devbot" in content


def test_local_run_wraps_user_message(tmp_path, monkeypatch):
    _clear_local_env(monkeypatch)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    agent = Agent(root=tmp_path, provider="local-vibethinker")
    monkeypatch.setattr(agent, "_stream_once", lambda: ("ok", None))
    monkeypatch.setattr(agent, "on_text", lambda text: None)

    result = agent.run("what do you think of devbot")

    assert result == "ok"
    assert agent.messages[1]["role"] == "user"
    assert agent.messages[1]["content"].startswith("/no_think")
    assert "what do you think of devbot" in agent.messages[1]["content"]


def test_local_run_does_not_wrap_when_thinking_enabled(tmp_path, monkeypatch):
    _clear_local_env(monkeypatch)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    agent = Agent(root=tmp_path, provider="local-vibethinker")
    agent.show_reasoning = True
    monkeypatch.setattr(agent, "_stream_once", lambda: ("ok", None))
    monkeypatch.setattr(agent, "on_text", lambda text: None)

    result = agent.run("what do you think of devbot")

    assert result == "ok"
    assert agent.messages[1]["content"] == "what do you think of devbot"
