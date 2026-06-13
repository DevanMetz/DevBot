"""The agentic loop: stream DeepSeek responses, execute tool calls, repeat."""

import json
import os
import time
from pathlib import Path

from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError

from .tools import TOOL_SCHEMAS, DANGEROUS_TOOLS, dispatch

# Transient errors worth retrying with backoff (vs. 4xx auth/balance errors).
RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)
MAX_ATTEMPTS = 4  # 1 initial try + 3 retries

# deepseek-chat / deepseek-reasoner share a 64K context window.
CONTEXT_LIMIT = 64_000
KNOWN_MODELS = {"deepseek-chat", "deepseek-reasoner"}


def _load_dotenv(root: Path):
    """Minimal .env loader (no dependency). Existing env vars win."""
    env_file = root / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"  # use "deepseek-reasoner" for harder tasks
DEFAULT_MAX_TURNS = 40  # tool-loop cap per message; override with DEVBOT_MAX_TURNS

SYSTEM_PROMPT = """\
You are DevBot, a CLI coding agent working in the user's project directory: {cwd}
Platform: {platform}

You help with software engineering tasks: writing code, fixing bugs, refactoring,
explaining code, and running commands.

Guidelines:
- Use tools to inspect the project before making claims about it. Read files before editing them.
- Prefer edit_file for small changes and write_file for new files or full rewrites.
- After making changes, verify them when possible (run tests, run the script, check syntax).
- Keep responses concise. Don't paste entire files back to the user; summarize what changed.
- If a command fails, read the error and fix the underlying problem rather than retrying blindly.
- Never run destructive commands (rm -rf, force push, etc.) without explaining why first.
"""


class Agent:
    def __init__(self, root: Path, model: str | None = None, auto_approve: bool = False):
        _load_dotenv(root)
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit(
                "DEEPSEEK_API_KEY is not set.\n"
                "Get a key at https://platform.deepseek.com and set it:\n"
                '  PowerShell:  $env:DEEPSEEK_API_KEY = "sk-..."\n'
                "  bash/zsh:    export DEEPSEEK_API_KEY=sk-..."
            )
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.model = model or os.environ.get("DEVBOT_MODEL", DEFAULT_MODEL)
        self.root = root
        self.auto_approve = auto_approve
        self.max_turns = int(os.environ.get("DEVBOT_MAX_TURNS", DEFAULT_MAX_TURNS))
        self.total_tokens = 0          # cumulative across the session
        self.last_prompt_tokens = 0    # prompt size of the most recent call
        self.messages: list[dict] = [
            {"role": "system",
             "content": SYSTEM_PROMPT.format(cwd=root, platform=os.name)}
        ]

    # ---- UI hooks (overridden/used by cli.py) -------------------------------
    def on_text(self, chunk: str):
        print(chunk, end="", flush=True)

    def on_tool_start(self, name: str, args: dict):
        preview = json.dumps(args)
        if len(preview) > 160:
            preview = preview[:160] + "..."
        print(f"\n\x1b[36m⏺ {name}\x1b[0m {preview}")

    def on_tool_end(self, result: str):
        first = result.splitlines()[0] if result else ""
        n = len(result.splitlines())
        print(f"\x1b[90m  ⎿ {first[:120]}{f' (+{n - 1} lines)' if n > 1 else ''}\x1b[0m")

    def confirm(self, name: str, args: dict) -> bool:
        if self.auto_approve:
            return True
        detail = args.get("command") or args.get("path", "")
        ans = input(f"\x1b[33m  Allow {name}({detail})? [y/N/a(lways)]\x1b[0m ").strip().lower()
        if ans == "a":
            self.auto_approve = True
            return True
        return ans == "y"

    # ---- core loop ----------------------------------------------------------
    def run(self, user_input: str):
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(self.max_turns):
            text, tool_calls = self._stream_once()

            msg: dict = {"role": "assistant", "content": text or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            self.messages.append(msg)

            if self.last_prompt_tokens > 0.85 * CONTEXT_LIMIT:
                print(f"\x1b[33m[devbot] Warning: prompt is {self.last_prompt_tokens:,} "
                      f"tokens, near the {CONTEXT_LIMIT:,} limit. Use /clear to reset.\x1b[0m")

            if not tool_calls:
                return  # model is done; final answer already streamed

            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                self.on_tool_start(name, args)

                if name in DANGEROUS_TOOLS and not self.confirm(name, args):
                    result = "User declined this tool call. Ask them how to proceed."
                else:
                    result = dispatch(name, args, self.root)

                self.on_tool_end(result)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

        print("\n[devbot] Reached max tool iterations for this message.")

    def _create_stream(self):
        """Open a streamed completion, retrying transient errors with backoff."""
        for attempt in range(MAX_ATTEMPTS):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=TOOL_SCHEMAS,
                    stream=True,
                    stream_options={"include_usage": True},
                )
            except RETRYABLE as e:
                if attempt == MAX_ATTEMPTS - 1:
                    raise
                wait = 2 ** attempt  # 1, 2, 4 seconds
                print(f"\x1b[33m[devbot] {type(e).__name__}; retrying in {wait}s "
                      f"(retry {attempt + 1}/{MAX_ATTEMPTS - 1})...\x1b[0m")
                time.sleep(wait)

    def _stream_once(self):
        """One streamed API call. Returns (text, tool_calls) with tool_calls
        assembled from streaming deltas into the standard dict shape."""
        stream = self._create_stream()
        text_parts: list[str] = []
        calls: dict[int, dict] = {}
        in_reasoning = False

        for chunk in stream:
            # Final chunk (include_usage) carries token counts and no choices.
            if getattr(chunk, "usage", None):
                self.last_prompt_tokens = chunk.usage.prompt_tokens
                self.total_tokens += chunk.usage.total_tokens
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # deepseek-reasoner streams chain-of-thought in reasoning_content
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                if not in_reasoning:
                    self.on_text("\x1b[90m")  # dim the thinking text
                    in_reasoning = True
                self.on_text(reasoning)
            elif in_reasoning:
                self.on_text("\x1b[0m\n")
                in_reasoning = False
            if delta.content:
                text_parts.append(delta.content)
                self.on_text(delta.content)
            for tc in delta.tool_calls or []:
                slot = calls.setdefault(tc.index, {
                    "id": "", "type": "function",
                    "function": {"name": "", "arguments": ""},
                })
                if tc.id:
                    slot["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        slot["function"]["name"] += tc.function.name
                    if tc.function.arguments:
                        slot["function"]["arguments"] += tc.function.arguments

        if in_reasoning:  # stream ended mid-reasoning; restore normal color
            self.on_text("\x1b[0m\n")
        if text_parts:
            self.on_text("\n")
        tool_calls = [calls[i] for i in sorted(calls)] or None
        return "".join(text_parts), tool_calls
