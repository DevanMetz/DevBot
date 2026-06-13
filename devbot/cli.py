"""Interactive REPL entry point: `devbot` or `python -m devbot`."""

import argparse
import sys
from pathlib import Path

# Enable arrow-key history/editing in the REPL when available. The stdlib
# `readline` ships on Linux/macOS; on Windows it comes from `pyreadline3`.
# Either way it auto-hooks input() on import, so we just try and move on.
try:
    import atexit
    import readline

    _HISTORY = Path.home() / ".devbot_history"
    try:
        readline.read_history_file(_HISTORY)
    except (FileNotFoundError, OSError):
        pass
    readline.set_history_length(1000)
    atexit.register(lambda: readline.write_history_file(_HISTORY))
except ImportError:
    pass

from openai import AuthenticationError, PermissionDeniedError

from . import __version__
from .agent import Agent, DEFAULT_MODEL, KNOWN_MODELS


def _friendly_api_error(e: Exception) -> str | None:
    """Turn common DeepSeek 4xx errors into actionable messages, else None."""
    if isinstance(e, AuthenticationError):
        return ("Authentication failed — your DEEPSEEK_API_KEY is missing or invalid. "
                "Check it at https://platform.deepseek.com/api_keys")
    if isinstance(e, PermissionDeniedError):
        return "Access denied by the API (check your account status/permissions)."
    msg = str(e).lower()
    if "insufficient balance" in msg:
        return ("Insufficient balance on your DeepSeek account — add credit at "
                "https://platform.deepseek.com")
    return None

COMMANDS_HELP = """\
Commands:
  /help          show this help
  /clear         reset the conversation (keeps system prompt)
  /stats         show token usage and message count
  /model <name>  switch model (deepseek-chat, deepseek-reasoner)
  /auto          toggle auto-approve of tool calls
  /exit          quit"""

BANNER = """\x1b[1m
  ____            ____        _
 |  _ \\  _____   _| __ )  ___ | |_
 | | | |/ _ \\ \\ / /  _ \\ / _ \\| __|
 | |_| |  __/\\ V /| |_) | (_) | |_
 |____/ \\___| \\_/ |____/ \\___/ \\__|  v{version}
\x1b[0m  DeepSeek-powered coding agent · model: {model} · cwd: {cwd}
  Type /help for commands.
"""


def main():
    parser = argparse.ArgumentParser(prog="devbot", description="DeepSeek-powered CLI coding agent")
    parser.add_argument("prompt", nargs="*", help="One-shot prompt (omit for interactive mode)")
    parser.add_argument("-m", "--model", default=None, help=f"Model id (default: {DEFAULT_MODEL})")
    parser.add_argument("-y", "--yes", action="store_true", help="Auto-approve all tool calls")
    parser.add_argument("-C", "--cwd", default=".", help="Project root to operate in")
    parser.add_argument("--version", action="version", version=f"devbot {__version__}")
    args = parser.parse_args()

    root = Path(args.cwd).resolve()
    agent = Agent(root=root, model=args.model, auto_approve=args.yes)

    if args.prompt:  # one-shot mode: devbot "fix the failing test"
        agent.run(" ".join(args.prompt))
        return

    print(BANNER.format(version=__version__, model=agent.model, cwd=root))
    while True:
        try:
            user = input("\x1b[1m\x1b[32m> \x1b[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            return
        if not user:
            continue
        if user in ("/exit", "/quit"):
            return
        if user == "/help":
            print(COMMANDS_HELP)
            continue
        if user == "/clear":
            agent.messages = agent.messages[:1]  # keep system prompt
            print("[conversation cleared]")
            continue
        if user == "/stats":
            print(f"[model: {agent.model} | last prompt: {agent.last_prompt_tokens:,} tokens "
                  f"| session total: {agent.total_tokens:,} tokens "
                  f"| messages: {len(agent.messages)} | max turns: {agent.max_turns} "
                  f"| auto-approve: {'on' if agent.auto_approve else 'off'}]")
            continue
        if user.startswith("/model"):
            parts = user.split(maxsplit=1)
            if len(parts) == 2:
                if parts[1] not in KNOWN_MODELS:
                    print(f"\x1b[33m[warning] '{parts[1]}' is not a known DeepSeek model "
                          f"({', '.join(sorted(KNOWN_MODELS))}). Setting it anyway.\x1b[0m")
                agent.model = parts[1]
            print(f"[model: {agent.model}]")
            continue
        if user == "/auto":
            agent.auto_approve = not agent.auto_approve
            print(f"[auto-approve: {'on' if agent.auto_approve else 'off'}]")
            continue
        try:
            agent.run(user)
        except KeyboardInterrupt:
            print("\n[interrupted]")
        except Exception as e:
            friendly = _friendly_api_error(e)
            msg = friendly or f"{type(e).__name__}: {e}"
            print(f"\x1b[31m[error] {msg}\x1b[0m", file=sys.stderr)


if __name__ == "__main__":
    main()
