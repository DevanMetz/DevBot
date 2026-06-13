"""Project-config file loader (.devbot/config.toml).

Precedence: real environment variables > .env file > config.toml > defaults.
"""

import os
import tomllib
from pathlib import Path

# Mapping from config.toml keys to their corresponding env var names.
_CONFIG_KEY_MAP: dict[str, str] = {
    "model": "DEVBOT_MODEL",
    "max_parallel": "DEVBOT_MAX_PARALLEL",
    "token_budget": "DEVBOT_TOKEN_BUDGET",
    "global_budget": "DEVBOT_GLOBAL_BUDGET",
    "loop_limit": "DEVBOT_LOOP_LIMIT",
    "compress_model": "DEVBOT_COMPRESS_MODEL",
    "mega_warn_threshold": "DEVBOT_MEGA_WARN_THRESHOLD",
    "pipeline_rounds": "DEVBOT_PIPELINE_ROUNDS",
}

# Temporary storage for config.toml values before they are applied to
# os.environ.  Populated by load_project_config(), flushed by
# apply_project_config().
_project_config: dict[str, str] = {}


def load_project_config(root: Path) -> None:
    """Load .devbot/config.toml into _project_config for later application.

    Call apply_project_config() after _load_dotenv() so that .env values
    take precedence over config.toml settings.

    Missing files and malformed TOML are silently ignored.
    """
    global _project_config
    _project_config.clear()
    config_path = root / ".devbot" / "config.toml"
    if not config_path.is_file():
        return
    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        # Malformed TOML — silently ignore (don't crash the agent)
        return
    if not isinstance(data, dict):
        return
    for key, env_var in _CONFIG_KEY_MAP.items():
        if key in data and env_var not in os.environ:
            _project_config[env_var] = str(data[key])


def apply_project_config() -> None:
    """Apply stored config.toml values to os.environ via setdefault.

    Real environment variables and values already loaded by _load_dotenv()
    take precedence — setdefault will not overwrite them.
    """
    for env_var, value in _project_config.items():
        os.environ.setdefault(env_var, value)
    _project_config.clear()
