"""Interactive REPL entry point: `devbot` or `python -m devbot`."""

import argparse
import sys
from pathlib import Path

# Enable arrow-key history/editing in the REPL when available. The stdlib
# `readline` ships on Linux/macOS; on Windows it comes from `pyreadline3`.
# Either way it auto-hooks input() on import, so we just try and move on.
def _setup_readline(history_path=None, marker_path=None):
    if history_path is None:
        history_path = Path.home() / ".devbot_history"
    if marker_path is None:
        marker_path = Path.home() / ".devbot_readline_warned"
    try:
        import atexit
        import readline

        try:
            readline.read_history_file(history_path)
        except (FileNotFoundError, OSError):
            pass
        readline.set_history_length(1000)
        try:
            atexit.register(lambda: readline.write_history_file(history_path))
        except Exception:
            pass
    except ImportError:
        if marker_path.exists():
            return
        print("[devbot] readline not available — pip install pyreadline3 to enable arrow-key history",
              file=sys.stderr)
        try:
            marker_path.write_text("")
        except OSError:
            pass

_setup_readline()

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
  /model <name>  switch model (deepseek-v4-flash, deepseek-v4-pro)
  /auto          toggle auto-approve of tool calls
  /think         toggle display of full chain-of-thought
  /swarm         toggle swarm mode (delegate to specialists; resets conversation)
  /megaswarm     toggle megaswarm mode (3 parallel agents + reviewer; resets conversation)
  /resume [id]   reload a saved session (latest if no id given)
  /sessions      list saved sessions with timestamps and token counts
  /exit          quit"""

BANNER = """\x1b[1m
  ____            ____        _
 |  _ \\  _____   _| __ )  ___ | |_
 | | | |/ _ \\ \\ / /  _ \\ / _ \\| __|
 | |_| |  __/\\ V /| |_) | (_) | |_
 |____/ \\___| \\_/ |____/ \\___/ \\__|  v{version}
\x1b[0m  DeepSeek-powered coding agent · model: {model}{mode} · cwd: {cwd}
  Type /help for commands.
"""


def main():
    parser = argparse.ArgumentParser(prog="devbot", description="DeepSeek-powered CLI coding agent")
    parser.add_argument("prompt", nargs="*", help="One-shot prompt (omit for interactive mode)")
    parser.add_argument("-m", "--model", default=None, help=f"Model id (default: {DEFAULT_MODEL})")
    parser.add_argument("-y", "--yes", action="store_true", help="Auto-approve all tool calls")
    parser.add_argument("-C", "--cwd", default=".", help="Project root to operate in")
    parser.add_argument("-s", "--swarm", action="store_true",
                        help="Swarm mode: manager agent can delegate to specialist sub-agents")
    parser.add_argument("-M", "--megaswarm", action="store_true",
                        help="Megaswarm mode: delegates 3 agents in parallel + reviewer synthesis")
    parser.add_argument("-r", "--resume", nargs="?", const="latest", metavar="ID",
                        help="Resume a saved session (latest if no ID given)")
    parser.add_argument("--run-plan", nargs="?", const="plan.md", default=None,
                        metavar="PLAN",
                        help="Autopilot: implement each '## Phase' of PLAN (default "
                             "plan.md) one at a time, verifying between phases. "
                             "Runs unattended (implies auto-approve).")
    parser.add_argument("--version", action="version", version=f"devbot {__version__}")
    args = parser.parse_args()

    root = Path(args.cwd).resolve()

    # Autopilot: run a whole plan unattended, phase by phase.
    if args.run_plan is not None:
        from .autopilot import run_plan
        print("\x1b[33m[devbot] Autopilot runs unattended with auto-approve and "
              "shell access. Ctrl+C to stop.\x1b[0m")
        ok = run_plan(root, args.run_plan, model=args.model)
        sys.exit(0 if ok else 1)

    # Handle --resume: restore from a saved session.
    if args.resume is not None:
        from .session import restore_agent, list_sessions
        session_id = None if args.resume == "latest" else args.resume
        restored = restore_agent(root, session_id=session_id,
                                 auto_approve=args.yes)
        if restored is None:
            if session_id:
                print(f"[devbot] Session '{session_id}' not found in {root / '.devbot'}", file=sys.stderr)
            else:
                print(f"[devbot] No saved sessions found in {root / '.devbot'}", file=sys.stderr)
            sys.exit(1)
        agent = restored
        print(f"[devbot] Resumed session {agent.session_id} "
              f"({agent.total_tokens:,} tokens, {len(agent.messages)} messages)")
    else:
        agent = Agent(root=root, model=args.model, auto_approve=args.yes,
                      swarm=args.swarm, megaswarm=args.megaswarm)

    if args.prompt:  # one-shot mode: devbot "fix the failing test"
        agent.run(" ".join(args.prompt))
        return

    print(BANNER.format(version=__version__, model=agent.model,
                        mode=" · megaswarm" if agent.megaswarm else (" · swarm" if agent.swarm else ""),
                        cwd=root))
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
            agent.total_tokens = 0
            agent.last_prompt_tokens = 0
            agent.delegation_count = 0
            if hasattr(agent, '_compressed'):
                agent._compressed = False
            print("[conversation cleared]")
            continue
        if user == "/stats":
            mode = "megaswarm" if agent.megaswarm else ("swarm" if agent.swarm else "solo")
            cost = agent.estimated_cost()
            line = (f"[model: {agent.model} | last prompt: {agent.last_prompt_tokens:,} tokens "
                    f"| session total: {agent.total_tokens:,} tokens "
                    f"| messages: {len(agent.messages)} | max turns: {agent.max_turns} "
                    f"| auto-approve: {'on' if agent.auto_approve else 'off'} "
                    f"| mode: {mode}"
                    f"{f' | delegations: {agent.delegation_count}' if agent.swarm or agent.megaswarm else ''}"
                    f" | cost: ${cost:.2f}")
            if agent.token_budget > 0:
                line += f" | budget: {agent.token_budget:,} tokens"
            print(line + "]")
            continue
        if user.startswith("/model"):
            parts = user.split(maxsplit=1)
            if len(parts) == 2:
                if parts[1] not in KNOWN_MODELS:
                    print(f"\x1b[33m[warning] '{parts[1]}' is not a known DeepSeek model "
                          f"({', '.join(sorted(KNOWN_MODELS))}). Setting it anyway.\x1b[0m")
                agent.model = parts[1]
                print(f"[model: {agent.model}]")
            else:
                print(f"[model: {agent.model}]  Usage: /model <name>  (current: {agent.model})")
            continue
        if user == "/auto":
            agent.auto_approve = not agent.auto_approve
            print(f"[auto-approve: {'on' if agent.auto_approve else 'off'}]")
            continue
        if user == "/think":
            agent.show_reasoning = not agent.show_reasoning
            print(f"[show reasoning: {'on' if agent.show_reasoning else 'off'}]")
            continue
        if user == "/swarm":
            # Recreate the agent with swarm toggled; this resets the conversation
            # because the system prompt and tool set change.
            # When downgrading from megaswarm, force swarm on instead of toggling off.
            if agent.megaswarm:
                enable = True
            else:
                enable = not agent.swarm
            agent = Agent(root=root, model=agent.model, auto_approve=agent.auto_approve,
                          swarm=enable, megaswarm=False)
            print(f"[swarm mode: {'on' if agent.swarm else 'off'} — conversation reset]")
            continue
        if user == "/megaswarm":
            # Toggle megaswarm on/off (megaswarm implies swarm).
            enable = not agent.megaswarm
            agent = Agent(root=root, model=agent.model, auto_approve=agent.auto_approve,
                          swarm=enable, megaswarm=enable)
            print(f"[megaswarm mode: {'on' if enable else 'off'} — conversation reset]")
            continue
        if user == "/sessions":
            from .session import list_sessions
            sessions = list_sessions(root)
            if not sessions:
                print("[no saved sessions]")
            else:
                print(f"[{len(sessions)} session(s) in {root / '.devbot'}]")
                for s in sessions:
                    mode = "megaswarm" if s["megaswarm"] else ("swarm" if s["swarm"] else "solo")
                    created = s["created"][:19] if s["created"] else "?"
                    sid_short = s["id"].replace("session-", "")[:17]
                    print(f"  {sid_short}  {created}  {s['model']:24s}  "
                          f"{s['total_tokens']:>10,}t  {s['message_count']:>4}msgs  {mode}")
            continue
        if user.startswith("/resume"):
            parts = user.split(maxsplit=1)
            sid = parts[1].strip() if len(parts) == 2 and parts[1].strip() else None
            from .session import restore_agent
            restored = restore_agent(root, session_id=sid, auto_approve=agent.auto_approve)
            if restored is None:
                print(f"[session not found: {sid or 'latest'}]")
            else:
                agent = restored
                print(f"[resumed session {agent.session_id} "
                      f"({agent.total_tokens:,} tokens, {len(agent.messages)} messages)]")
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
