"""The agentic loop: stream DeepSeek responses, execute tool calls, repeat."""

import json
import os
import sys
import threading
import time
from pathlib import Path

import httpx

# Serializes interactive approval prompts so parallel megaswarm sub-agents
# can't race on stdin and corrupt each other's input.
_CONFIRM_LOCK = threading.Lock()

from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError

from .tools import TOOL_SCHEMAS, DANGEROUS_TOOLS, ALLOW_LIST, dispatch, check_command
from .devlog import log_tool_call, log_turn

# Transient errors worth retrying with backoff (vs. 4xx auth/balance errors).
RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)
MAX_ATTEMPTS = 4  # 1 initial try + 3 retries

# DeepSeek V4 models have a 1M-token context window.
CONTEXT_LIMIT = 1_000_000
# deepseek-chat / deepseek-reasoner are legacy aliases for V4-flash (non-thinking /
# thinking); they are deprecated 2026-07-24 but still valid until then.
KNOWN_MODELS = {"deepseek-v4-pro", "deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner"}

# ---------------------------------------------------------------------------
# Session-wide (process-level) token budget — shared across all agents in
# autopilot / megaswarm runs.
# ---------------------------------------------------------------------------
_GLOBAL_TOKEN_COUNT = 0
_GLOBAL_BUDGET = int(os.environ.get("DEVBOT_GLOBAL_BUDGET", "0") or "0")


def reset_global_token_count() -> None:
    """Zero the global token counter (useful in tests)."""
    global _GLOBAL_TOKEN_COUNT
    _GLOBAL_TOKEN_COUNT = 0


def get_global_token_count() -> int:
    """Return the current global token count."""
    return _GLOBAL_TOKEN_COUNT


def check_global_budget_exceeded() -> bool:
    """Return True iff the global budget is set (>0) and has been reached."""
    return _GLOBAL_BUDGET > 0 and _GLOBAL_TOKEN_COUNT >= _GLOBAL_BUDGET


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
DEFAULT_MODEL = "deepseek-v4-flash"  # use "deepseek-v4-pro" for harder tasks
DEFAULT_MAX_TURNS = 200  # tool-loop cap per message; override with DEVBOT_MAX_TURNS

# Per-1M-token pricing (USD), cache-miss input rates per the DeepSeek pricing
# page. deepseek-chat / deepseek-reasoner are legacy aliases for V4-flash and
# share its pricing. Cost estimates are approximate (cache hits cost less).
MODEL_PRICING = {
    "deepseek-v4-flash":  {"input": 0.14, "output": 0.28, "cache_hit_input": 0.014},
    "deepseek-v4-pro":    {"input": 0.435, "output": 0.87, "cache_hit_input": 0.0435},
    "deepseek-chat":      {"input": 0.14, "output": 0.28, "cache_hit_input": 0.014},
    "deepseek-reasoner":  {"input": 0.14, "output": 0.28, "cache_hit_input": 0.014},
}

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

MANAGER_ADDENDUM = """\

You are running as the MANAGER of an agent swarm. In addition to your own tools,
you have a `delegate(role, task)` tool that hands a self-contained subtask to a
specialist sub-agent (coder, reviewer, tester, researcher) and returns its result.

- Break larger requests into clear subtasks and delegate them; give each specialist
  enough context to work without seeing this conversation.
- Use `reviewer`/`researcher` for read-only analysis, `coder` for edits, `tester` for tests.
- Do simple things yourself; delegate when a task benefits from focus or specialization.
- Synthesize specialists' results into a clear final answer for the user.

You also have `pipeline(task)`: a coder implements the task, a reviewer audits the
actual files for real bugs, and the coder fixes any findings (looping until clean).

HARD RULE — code changes MUST go through review:
- For ANY task that creates, edits, or deletes files, you MUST use `pipeline(task)`.
- Do NOT call `write_file`, `edit_file`, or mutating `run_command`s yourself, and do
  NOT use a bare `delegate("coder", ...)` to make changes — those skip review and
  ship bugs. `pipeline` is the only sanctioned way to modify code.
- Your own `read_file`/`grep`/`find_files`/`list_dir` are fine for understanding the
  task and for splitting it into pipeline calls.
- For a large change, break it into a sequence of `pipeline(task)` calls (one per
  coherent unit); each gets its own review→fix loop.
- Only `delegate`/`megadelegate` for READ-ONLY work (analysis, research, review).
"""

MEGASWARM_ADDENDUM = """\

You are running as the MANAGER of a MEGASWARM. In addition to your own tools and
the normal `delegate(role, task)` tool, you also have `megadelegate(task, n=5, role=None)` which:

- Launches N specialists (default 3) IN PARALLEL on the SAME task.
- If `role` is given, all N specialists share that role; otherwise the default trio
  (coder, researcher, tester) is cycled for N positions.
- A reviewer then combines their independent outputs into one synthesis.

The HARD RULE above still applies: code changes go through `pipeline`. Use the
megaswarm tools as follows:

- `pipeline(task)` — the default for ANY code change (review→fix enforced). For a
  big change, issue several `pipeline` calls in sequence.
- `megadelegate(task)` (no subtasks) — TRIANGULATED ANALYSIS of one question
  (read-only); the agents investigate and a reviewer synthesises. Never for edits.
- `megadelegate(task, subtasks=[...])` — only when you have MANY genuinely
  INDEPENDENT units touching DIFFERENT files and want them built concurrently.
  Its agents edit in parallel with a single integrating reviewer (NOT a per-unit
  fix loop), so it trades some review depth for speed — after it finishes, run a
  `pipeline` pass over anything correctness-critical. Never put interdependent
  same-file edits in separate subtasks (they conflict).
"""


class Agent:
    def __init__(self, root: Path, model: str | None = None, auto_approve: bool = False,
                 system_prompt: str | None = None, tool_schemas: list | None = None,
                 swarm: bool = False, megaswarm: bool = False,
                 label: str | None = None):
        _load_dotenv(root)
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit(
                "DEEPSEEK_API_KEY is not set.\n"
                "Get a key at https://platform.deepseek.com and set it:\n"
                '  PowerShell:  $env:DEEPSEEK_API_KEY = "sk-..."\n'
                "  bash/zsh:    export DEEPSEEK_API_KEY=sk-..."
            )
        self.client = OpenAI(
            api_key=api_key, base_url=DEEPSEEK_BASE_URL,
            http_client=httpx.Client(
                limits=httpx.Limits(max_connections=64,
                                    max_keepalive_connections=32)),
            timeout=httpx.Timeout(120.0, connect=10.0))
        self.model = model or os.environ.get("DEVBOT_MODEL", DEFAULT_MODEL)
        self.root = root
        self.auto_approve = auto_approve
        self.swarm = swarm
        self.megaswarm = megaswarm
        self.label = label  # output prefix for sub-agents, e.g. "coder"
        self.max_turns = int(os.environ.get("DEVBOT_MAX_TURNS", DEFAULT_MAX_TURNS))
        self.total_tokens = 0          # cumulative across the session
        self.completion_tokens = 0     # cumulative output tokens
        self.prompt_cache_hit_tokens = 0   # cumulative cache-hit input tokens
        self.prompt_cache_miss_tokens = 0  # cumulative cache-miss input tokens
        self.last_prompt_tokens = 0    # prompt size of the most recent call
        self.delegation_count = 0      # number of specialist delegations (swarm)
        self._compressed = False       # suppress repeated context-warning spam
        self.session_id: str | None = None  # set by session.save_session on first save
        self.show_reasoning = os.environ.get("DEVBOT_SHOW_REASONING", "").lower() in ("1", "true", "yes")
        self.allow_shell = os.environ.get("DEVBOT_ALLOW_SHELL", "").lower() in ("1", "true")
        self.token_budget = int(os.environ.get("DEVBOT_TOKEN_BUDGET", "0") or "0")
        self._budget_exhausted = False
        self.loop_limit = int(os.environ.get("DEVBOT_LOOP_LIMIT", "3"))
        self._last_call_sig = None      # tuple of (name, args_str) for loop detection
        self._same_call_count = 0
        self._last_error = None         # last error string for error-loop detection
        self._same_error_count = 0

        self.tool_schemas = list(tool_schemas) if tool_schemas is not None else list(TOOL_SCHEMAS)
        if swarm or megaswarm:  # the manager gets the delegate + pipeline tools
            from .swarm import delegate_schema, pipeline_schema
            self.tool_schemas.append(delegate_schema())
            self.tool_schemas.append(pipeline_schema())
        if megaswarm:
            from .swarm import megadelegate_schema
            self.tool_schemas.append(megadelegate_schema())

        if system_prompt is not None:
            prompt = system_prompt
        else:
            prompt = SYSTEM_PROMPT.format(cwd=root, platform=os.name)
            if megaswarm:
                prompt += MANAGER_ADDENDUM + MEGASWARM_ADDENDUM
            elif swarm:
                prompt += MANAGER_ADDENDUM
        self.messages: list[dict] = [{"role": "system", "content": prompt}]

    # ---- UI hooks (overridden/used by cli.py) -------------------------------
    @staticmethod
    def _transient_line(text: str, clear: bool = False):
        """Print/update a single in-place transient line (no scroll).

        When *clear* is True, erase the transient line entirely.
        When *clear* is False, write *text* on the current terminal line
        using a carriage return + clear-to-end-of-line.
        """
        # Gracefully handle terminals that can't encode the text.
        try:
            encoded = text.encode(sys.stdout.encoding or 'utf-8', errors='replace')
            safe = encoded.decode(sys.stdout.encoding or 'utf-8')
        except (UnicodeError, LookupError):
            safe = text.encode('ascii', errors='replace').decode('ascii')
        if clear:
            print("\r\x1b[2K", end="", flush=True)
        else:
            print(f"\r{safe}\x1b[K", end="", flush=True)

    @staticmethod
    def _safe_print(*args, **kwargs):
        """print() that survives terminals that can't encode Unicode."""
        try:
            print(*args, **kwargs)
        except UnicodeEncodeError:
            # Replace non-encodable characters with '?'
            safe_args = []
            for a in args:
                if isinstance(a, str):
                    a = a.encode(sys.stdout.encoding or 'utf-8',
                                  errors='replace').decode(sys.stdout.encoding or 'utf-8')
                safe_args.append(a)
            print(*safe_args, **kwargs)

    def on_text(self, chunk: str):
        self._safe_print(chunk, end="", flush=True)

    def on_tool_start(self, name: str, args: dict):
        preview = json.dumps(args)
        if len(preview) > 160:
            preview = preview[:160] + "..."
        tag = f"\x1b[35m[{self.label}]\x1b[0m " if self.label else ""
        self._safe_print(f"\n{tag}\x1b[36m>> {name}\x1b[0m {preview}")

    def on_tool_end(self, result: str):
        first = result.splitlines()[0] if result else ""
        n = len(result.splitlines())
        self._safe_print(f"\x1b[90m  = {first[:120]}{f' (+{n - 1} lines)' if n > 1 else ''}\x1b[0m")

    def confirm(self, name: str, args: dict, reason: str = "") -> bool:
        if self.auto_approve:
            return True
        detail = args.get("command") or args.get("path", "")
        if reason:
            detail += f" — {reason}"
        tag = f"[{self.label}] " if self.label else ""
        with _CONFIRM_LOCK:  # one prompt at a time, even across parallel sub-agents
            ans = input(f"\x1b[33m  {tag}Allow {name}({detail})? [y/N/a(lways)]\x1b[0m ").strip().lower()
        if ans == "a":
            self.auto_approve = True
            return True
        return ans == "y"

    # ---- core loop ----------------------------------------------------------
    def _auto_save(self):
        """Persist the current session (main agent only)."""
        if self.label is None:
            from .session import save_session
            save_session(self)

    def _check_error_loop(self, result: str) -> bool:
        """Check for repeated identical errors.  Returns True if the agent
        should halt because the same error has been seen *loop_limit* times
        in a row."""
        if self.loop_limit <= 0:
            return False
        if not result.startswith("Error"):
            # Non-error result breaks any error streak.
            self._last_error = None
            self._same_error_count = 0
            return False
        if result == self._last_error:
            self._same_error_count += 1
            if self._same_error_count >= self.loop_limit:
                self._safe_print(
                    f"[devbot] Stuck error loop detected: same error "
                    f"repeated {self._same_error_count} times. Halting."
                )
                return True
        else:
            self._same_error_count = 1
            self._last_error = result
        return False

    def estimated_cost(self) -> float:
        """Estimated USD cost of the session.

        Uses real input/output/cache-hit breakdown when available from the API;
        falls back to a 50/50 input/output split heuristic when only total
        tokens are tracked. If the model isn't in the pricing table, defaults
        to deepseek-v4-flash prices.
        """
        pricing = MODEL_PRICING.get(self.model, MODEL_PRICING["deepseek-v4-flash"])
        input_price = pricing["input"] / 1_000_000
        output_price = pricing["output"] / 1_000_000
        cache_hit_input_price = pricing.get("cache_hit_input", input_price) / 1_000_000

        # If we have a real usage breakdown, use it
        completion = getattr(self, "completion_tokens", 0)
        cache_hit = getattr(self, "prompt_cache_hit_tokens", 0)
        cache_miss = getattr(self, "prompt_cache_miss_tokens", 0)
        if completion > 0 or cache_hit > 0 or cache_miss > 0:
            # Use the detailed breakdown
            output_cost = completion * output_price
            cache_hit_cost = cache_hit * cache_hit_input_price
            cache_miss_cost = cache_miss * input_price
            return cache_hit_cost + cache_miss_cost + output_cost

        # Fallback: 50/50 split heuristic (no breakdown available)
        half = self.total_tokens / 2
        return half * input_price + half * output_price

    def run(self, user_input: str) -> str:
        """Run the agentic loop until the model stops calling tools.

        Returns the model's final assistant text (used as the result when this
        agent is a delegated specialist)."""
        # Reset per-run loop-detection state so counters don't leak across
        # successive invocations (e.g. the CLI REPL calls run() repeatedly).
        self._last_call_sig = None
        self._same_call_count = 0
        self._last_error = None
        self._same_error_count = 0

        self.messages.append({"role": "user", "content": user_input})

        if check_global_budget_exceeded():
            self._safe_print(
                f"[devbot] Global token budget exhausted "
                f"({get_global_token_count():,} >= {_GLOBAL_BUDGET:,}). "
                f"Process halted."
            )
            self._auto_save()
            return ""

        if self.token_budget > 0 and self.total_tokens >= self.token_budget:
            self._safe_print(
                f"[devbot] Token budget exhausted ({self.total_tokens:,} >= "
                f"{self.token_budget:,}). Session halted."
            )
            self._auto_save()
            return ""

        for _ in range(self.max_turns):
            text, tool_calls = self._stream_once()

            # --- stuck-loop detection (identical tool calls) ---
            if self.loop_limit > 0 and tool_calls:
                sig = tuple((tc["function"]["name"],
                             json.dumps(
                                 json.loads(tc["function"]["arguments"] or "{}"),
                                 sort_keys=True))
                             for tc in tool_calls)
                if sig == self._last_call_sig:
                    self._same_call_count += 1
                    if self._same_call_count >= self.loop_limit:
                        self._safe_print(
                            f"[devbot] Stuck loop detected: same tool call "
                            f"repeated {self._same_call_count} times. Halting."
                        )
                        self._auto_save()
                        return text or ""
                else:
                    self._same_call_count = 1
                    self._last_call_sig = sig

            if self._budget_exhausted:
                if check_global_budget_exceeded():
                    self._safe_print(
                        f"[devbot] Global token budget exhausted "
                        f"({get_global_token_count():,} >= {_GLOBAL_BUDGET:,}). "
                        f"Session halted."
                    )
                else:
                    self._safe_print(
                        f"[devbot] Token budget exhausted ({self.total_tokens:,} >= "
                        f"{self.token_budget:,}). Session halted."
                    )
                self._auto_save()
                return text or ""

            msg: dict = {"role": "assistant", "content": text or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            self.messages.append(msg)

            if not tool_calls:
                self._auto_save()
                return text or ""  # model is done; final answer already streamed

            # Remember where the assistant message was so we can roll back on Ctrl+C.
            assistant_idx = len(self.messages) - 1
            try:
                for tc in tool_calls:
                    name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        raw_args = tc["function"]["arguments"]
                        result = (f"Error: failed to parse tool arguments as JSON. "
                                  f"Raw arguments string: {raw_args!r}")
                        if self._check_error_loop(result):
                            self._auto_save()
                            return text or ""
                        if tc["id"]:
                            self.messages.append({
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": result,
                            })
                        continue

                    if name == "delegate":
                        self.on_tool_start(name, args)
                        from .swarm import run_specialist
                        result = run_specialist(self, args.get("role", ""), args.get("task", ""))
                    elif name == "megadelegate":
                        self.on_tool_start(name, args)
                        from .swarm import run_megaswarm
                        result = run_megaswarm(self, args.get("task", ""),
                                               n=int(args.get("n", 3)),
                                               role=args.get("role"),
                                               subtasks=args.get("subtasks"))
                    elif name == "pipeline":
                        self.on_tool_start(name, args)
                        from .swarm import run_pipeline
                        result = run_pipeline(self, args.get("task", ""))
                    elif name == "run_command":
                        status, reason = check_command(args.get("command", ""))
                        if status == "blocked":
                            result = f"Error: command blocked — {reason}"
                        elif status == "allowed":
                            self.on_tool_start(name, args)
                            result = dispatch(name, args, self.root)
                        else:  # needs_approval
                            if self.allow_shell and self.auto_approve:
                                self.on_tool_start(name, args)
                                result = dispatch(name, args, self.root)
                            elif not self.confirm(name, args, reason=reason):
                                result = "User declined this tool call. Ask them how to proceed."
                            else:
                                self.on_tool_start(name, args)
                                result = dispatch(name, args, self.root)
                    elif name in DANGEROUS_TOOLS and not self.confirm(name, args):
                        result = "User declined this tool call. Ask them how to proceed."
                    else:
                        self.on_tool_start(name, args)
                        result = dispatch(name, args, self.root)

                    if self._check_error_loop(result):
                        self._auto_save()
                        return text or ""

                    log_tool_call(name, args, result, ok=not result.startswith("Error"))
                    self.on_tool_end(result)
                    if not tc["id"]:
                        print(f"\x1b[33m[devbot] Warning: tool_call_id is empty for "
                              f"tool '{name}'; skipping tool result.\x1b[0m")
                        continue
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
            except KeyboardInterrupt:
                # Roll back the assistant message and any partial tool results
                # so the conversation state stays consistent (no orphaned
                # assistant messages with unresolved tool_calls).
                del self.messages[assistant_idx:]
                self._auto_save()
                raise

        print("\n[devbot] Reached max tool iterations for this message.")
        self._auto_save()
        return ""

    def _create_stream(self):
        """Open a streamed completion, retrying transient errors with backoff."""
        for attempt in range(MAX_ATTEMPTS):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tool_schemas,
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
        finish_reason = None

        for chunk in stream:
            # Final chunk (include_usage) carries token counts and no choices.
            if getattr(chunk, "usage", None):
                usage = chunk.usage
                self.last_prompt_tokens = usage.prompt_tokens
                self.total_tokens += usage.total_tokens
                # Accumulate usage breakdown for accurate cost estimation
                ct = getattr(usage, "completion_tokens", 0)
                if isinstance(ct, int):
                    self.completion_tokens += ct
                if hasattr(usage, "prompt_cache_hit_tokens"):
                    ht = getattr(usage, "prompt_cache_hit_tokens", 0)
                    mt = getattr(usage, "prompt_cache_miss_tokens", 0)
                    if isinstance(ht, int):
                        self.prompt_cache_hit_tokens += ht
                    if isinstance(mt, int):
                        self.prompt_cache_miss_tokens += mt
                global _GLOBAL_TOKEN_COUNT
                _GLOBAL_TOKEN_COUNT += usage.total_tokens
                if self.token_budget > 0 and self.total_tokens >= self.token_budget:
                    self._budget_exhausted = True
                if check_global_budget_exceeded():
                    self._budget_exhausted = True
                log_turn(self.model, self.last_prompt_tokens, self.total_tokens)
                # P1-1: context threshold check, now inside _stream_once right
                # after the usage chunk so it fires at most once per API call.
                if (self.last_prompt_tokens > 0.85 * CONTEXT_LIMIT
                        and not self._compressed):
                    print(f"\x1b[33m[devbot] Warning: prompt is {self.last_prompt_tokens:,} "
                          f"tokens, near the {CONTEXT_LIMIT:,} limit.\x1b[0m")
                    self._compressed = True  # suppress until next successful compression
                    if self._compress_conversation():
                        print("\x1b[90m[devbot] Conversation compressed to free "
                              "context space.\x1b[0m")
                        self._compressed = False
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # P1-7: capture finish_reason from the final choice chunk
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            # thinking mode streams chain-of-thought in reasoning_content
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                if not in_reasoning and not self.show_reasoning:
                    # Start of CoT: show compact thinking indicator
                    self._reasoning_chars = 0
                    self._transient_line("💭 thinking\u2026 (0 tokens)")
                if self.show_reasoning and not in_reasoning:
                    self.on_text("\x1b[90m")
                in_reasoning = True
                if self.show_reasoning:
                    self.on_text(reasoning)
                else:
                    self._reasoning_chars += len(reasoning)
                    est_tokens = max(1, self._reasoning_chars // 4)
                    self._transient_line(
                        f"💭 thinking\u2026 ({est_tokens} tokens)")
                if delta.content:
                    if self.show_reasoning:
                        self.on_text("\x1b[0m\n")
                    else:
                        self._transient_line(clear=True)
                    in_reasoning = False
                    text_parts.append(delta.content)
                    self.on_text(delta.content)
            else:
                if in_reasoning:
                    self.on_text("\x1b[0m\n")
                    in_reasoning = False
                if delta.content:
                    text_parts.append(delta.content)
                    self.on_text(delta.content)
            # P1-16: reset ANSI color before tool-call deltas if still in
            # reasoning (prevents colour bleed with thinking models).
            if delta.tool_calls and in_reasoning:
                if self.show_reasoning:
                    self.on_text("\x1b[0m\n")
                else:
                    self._transient_line(clear=True)
                in_reasoning = False
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

        if in_reasoning:  # stream ended mid-reasoning; clean up display
            if self.show_reasoning:
                self.on_text("\x1b[0m\n")
            else:
                self._transient_line(clear=True)
        if text_parts:
            self.on_text("\n")
        # P1-7: warn if the model stopped because it hit the output length limit
        if finish_reason == "length":
            print("\x1b[33m[devbot] Warning: response truncated "
                  "(finish_reason=length).\x1b[0m")
        tool_calls = [calls[i] for i in sorted(calls)] or None
        return "".join(text_parts), tool_calls

    def _keep_index(self) -> int:
        """Index of the last user message — the start of the current turn.

        Everything from here on (assistant + its tool_calls + tool results) must
        be kept together; splitting it would orphan a tool message and the API
        would reject the next call. Falls back to keeping the last 2 messages."""
        for i in range(len(self.messages) - 1, 0, -1):
            if self.messages[i]["role"] == "user":
                return i
        return max(1, len(self.messages) - 2)

    def _compress_conversation(self) -> bool:
        """Summarize older messages to free context space. Returns True if compressed.

        Preserves the system prompt and the entire in-progress turn (from the last
        user message onward) so assistant/tool_call pairings stay intact."""
        keep = self._keep_index()
        to_compress = self.messages[1:keep]
        if len(to_compress) <= 1:
            return False  # nothing meaningful to compress yet

        tail = self.messages[keep:]
        summary_prompt = (
            "Summarise the following conversation history concisely. "
            "Capture key decisions, facts learned, files examined, changes made, "
            "and any unresolved issues. Be brief but complete."
        )
        compress_msgs = [
            {"role": "system", "content": "You are a concise conversation summariser."},
            {"role": "user", "content": f"{summary_prompt}\n\nConversation:\n{json.dumps(to_compress, separators=(',', ':'))}"},
        ]

        try:
            compress_model = os.environ.get("DEVBOT_COMPRESS_MODEL", "deepseek-v4-flash")
            resp = self.client.chat.completions.create(
                model=compress_model,
                messages=compress_msgs,
                stream=False,
                timeout=60,
            )
            summary = resp.choices[0].message.content
            if resp.usage:
                self.total_tokens += resp.usage.total_tokens
                global _GLOBAL_TOKEN_COUNT
                _GLOBAL_TOKEN_COUNT += resp.usage.total_tokens
            if not summary:
                self.messages = [self.messages[0]] + tail  # truncate, keep turn intact
                return True
        except Exception:
            self.messages = [self.messages[0]] + tail  # truncate on failure
            return True

        self.messages = (
            [self.messages[0]]
            + [{"role": "system", "content": f"[Compressed conversation history]\n{summary}"}]
            + tail
        )
        return True
