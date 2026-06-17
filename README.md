# DevBot

A Claude Code-style CLI coding agent powered by OpenAI-compatible chat APIs.

DevBot runs an agentic loop in your terminal: you describe a task, the model reads
your code with tools, edits files, and runs commands (with your approval) until the
task is done.

## Features

- **Streaming agentic loop** with OpenAI-compatible function calling against DeepSeek or a local llama.cpp server.
- **Twelve built-in tools** for reading, searching, editing, running, web-searching, and testing code.
- **Pipeline mode** — mandatory review→fix loop so every code change is audited before it ships.
- **Session persistence** — sessions auto-save after each turn; resume later with `--resume` or `/resume`.
- **Autopilot** — `--run-plan` reads `plan.md` and implements each `## Phase` step-by-step, unattended.
- **Cost estimation** — live USD cost estimate based on per-model pricing.
- **Structured JSONL logging** — set `DEVBOT_LOG` to a file path for timestamped tool-call and turn logs.
- **Global token budget** — process-wide cap (`DEVBOT_GLOBAL_BUDGET`) shared across all swarm agents.
- **Token-efficiency mode** — compact prompts, smaller tool returns, and clipped agent handoffs for long runs.
- **Approval gates** — writes, shell commands, and test runs require confirmation (`y` / `a`lways / decline).
- **Sandboxed file access** — tools cannot read or write outside the project root.
- **Swarm mode** — a manager agent can delegate subtasks to specialist sub-agents.
- **Web search** — `web_search` finds docs and examples via DuckDuckGo.
- **Reasoning support** — thinking-mode chain-of-thought is streamed dimmed.
- **Resilient** — transient API errors are retried with exponential backoff; auth/balance
  errors give clear, actionable messages.
- **Token tracking** with a context-window warning and auto-compression, plus REPL history and `.env` support.

## Tools available to the model

| Tool | What it does | Needs approval |
|---|---|---|
| `read_file` | Read a file with line numbers | no |
| `list_dir` | List a directory | no |
| `grep` | Regex search across the project (supports `**` recursive patterns) | no |
| `find_files` | Find files by name glob (e.g. `*.py`, supports `**` recursive patterns) | no |
| `web_search` | Web search via DuckDuckGo (titles, URLs, snippets) | no |
| `write_file` | Create/overwrite a file | yes |
| `edit_file` | Exact string replacement in a file | yes |
| `run_command` | Shell command in the project root | yes |
| `verify` | Auto-detect & run the project's test suite | yes |
| `delegate` | Delegate a subtask to a specialist sub-agent (swarm only) | no |
| `pipeline` | Implement a task with automatic review→fix loop (swarm only) | no |
| `megadelegate` | Launch N specialists in parallel + reviewer synthesis (megaswarm only) | no |

## Setup

```sh
pip install -e .
```

This installs the `openai` SDK and `ddgs` (DuckDuckGo search, used by `web_search`).

On Windows, optionally install `pyreadline3` for arrow-key history and line editing:
```sh
pip install -e .[win]
```
(Or `pip install pyreadline3`.)

Get an API key at https://platform.deepseek.com, then:

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."   # PowerShell
# export DEEPSEEK_API_KEY=sk-...   # bash/zsh
```

### Local VibeThinker

DevBot can use the local VibeThinker-3B llama.cpp/Vulkan server as another
OpenAI-compatible provider.

Start the model server:

```powershell
& "C:\Users\Devan\Documents\Codex\2026-06-16\hey-how-can-i-run-this\outputs\vibethinker-vulkan\start-vibethinker-q4-vulkan.ps1"
```

Then start DevBot with the local provider:

```sh
devbot --local
devbot --local "fix the failing test"
```

Inside the REPL, you can switch providers without restarting:

```text
/local
/cloud
```

The default local connection is `http://127.0.0.1:8092/v1` with model
`vibethinker-q4-vulkan`, API key `local`, and a 600-token local output cap.
For custom local servers, set:

```powershell
$env:LOCAL_LLM_BASE_URL = "http://127.0.0.1:8092/v1"
$env:LOCAL_LLM_MODEL = "vibethinker-q4-vulkan"
$env:LOCAL_LLM_API_KEY = "local"
$env:LOCAL_LLM_MAX_TOKENS = "600"
```

You can check the server with:

```sh
python -m devbot.local_llm_health
# or, after reinstalling with pip install -e .
devbot-local-health
```

## Usage

```sh
devbot                       # interactive REPL in the current directory
devbot "fix the failing test"  # one-shot mode
devbot -y                    # auto-approve all tool calls (be careful)
devbot -m deepseek-v4-pro    # use the more capable model for harder tasks
devbot -C path/to/project    # operate on another directory
devbot -s                    # swarm mode
devbot -M                    # megaswarm mode (3 parallel agents + reviewer)
devbot --local               # use local VibeThinker instead of DeepSeek
devbot --run-plan            # autopilot: implement each `## Phase` from plan.md
devbot --run-plan my-plan.md # use a custom plan file
devbot --resume              # resume the latest saved session
devbot --resume <session-id> # resume a specific session
```

In the REPL:

- `/help` — list commands
- `/clear` — reset the conversation (keeps system prompt)
- `/stats` — show token usage (last prompt + session total) and message count
- `/model <name>` — switch model (`deepseek-v4-flash` or `deepseek-v4-pro`; warns on unknown names)
- `/local` — switch to VibeThinker Local (resets the conversation)
- `/cloud` — switch back to DeepSeek (resets the conversation)
- `/auto` — toggle auto-approve
- `/think` — toggle display of chain-of-thought reasoning
- `/swarm` — toggle swarm mode (resets the conversation)
- `/megaswarm` — toggle megaswarm mode (3 parallel agents + reviewer; resets conversation)
- `/resume [id]` — reload a saved session (latest if no id given)
- `/sessions` — list saved sessions with timestamps and token counts
- `/tools` — list available tool names
- `/cost` — show estimated session cost and token total
- `/exit` — quit

DevBot tracks token usage per call and warns (and auto-compresses) when a prompt
approaches the 1M context limit. Set `DEVBOT_MAX_TURNS` to cap tool iterations
per message (default 200).

When the model wants to write a file or run a command you'll get a prompt:
`y` allows once, `a` allows everything for the rest of the session, anything
else declines (the model is told and can adjust).

## Swarm mode

Run with `--swarm` (or `-s`, or toggle with `/swarm` in the REPL) to make the
agent a **manager** that can delegate subtasks to specialist sub-agents:

```sh
devbot --swarm "add a config loader with tests and review it"
```

The manager gets a `delegate(role, task)` tool. Each call spawns a fresh sub-agent
with its own system prompt, restricted tool set, and agentic loop; its final answer
is returned to the manager, which synthesizes the results.

| Specialist | Tools | Purpose |
|---|---|---|
| `coder` | all tools | write, edit, refactor code |
| `reviewer` | read-only (incl. `web_search`) | find bugs, edge cases, style issues |
| `tester` | read + `run_command` + `verify` | run tests, diagnose failures |
| `researcher` | read-only (incl. `web_search`) | explore the codebase, answer questions |

Sub-agents never get the `delegate` tool themselves (no recursion) and still
respect approval gates. Specialist definitions live in
[`devbot/swarm.py`](devbot/swarm.py).

### Pipeline mode

`pipeline(task)` is the sanctioned way to make code changes in swarm mode.
It runs a coder → reviewer → coder-fix loop:

1. A `coder` implements the task.
2. A `reviewer` inspects the actual files for real bugs and responds with
   `VERDICT: CLEAN` or `VERDICT: ISSUES` (with a numbered list of concrete fixes).
3. If issues are found, the coder fixes them and the cycle repeats (up to the
   round limit set by `DEVBOT_PIPELINE_ROUNDS`, default 2).

The manager **must** use `pipeline` for any task that creates, edits, or deletes
files. Direct use of `write_file`, `edit_file`, or mutating `run_command` by the
manager is prohibited — this is enforced by a **HARD RULE** in the system prompt.

## Megaswarm mode

Run with `--megaswarm` (or `-M`, or toggle with `/megaswarm`) for a heavier
pattern. The manager additionally gets a `megadelegate(task)` tool that:

1. Launches the trio (`coder`, `researcher`, `tester`) **in parallel** on the same task.
2. Hands their three independent outputs to a `reviewer` that synthesizes one combined answer.

```sh
devbot --megaswarm "refactor the auth module to async"
```

Because the trio runs concurrently, their per-token streaming is silenced during
the parallel phase (you get a clean per-agent status line instead); the reviewer's
synthesis streams normally. Approval prompts are serialized across threads.

> **Tip:** run megaswarm with `-y` (auto-approve). With approval on, the parallel
> `coder` will block waiting for confirmation, which is awkward mid-parallel-run.

### Divide-and-conquer mode

`megadelegate(task, subtasks=[...])` enables a divide-and-conquer pattern:
you pass a list of independent subtasks, each gets its own coder, and a reviewer
integrates the results.

- Each subtask runs concurrently — use for genuinely independent units touching
  **different files**.
- Do **not** put interdependent edits to the same file in separate subtasks
  (they will conflict).
- After a divide-and-conquer `megadelegate`, run a `pipeline` pass over anything
  correctness-critical for an extra safety net.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API key (required for the default provider) | — |
| `DEVBOT_PROVIDER` | LLM provider (`deepseek` or `local-vibethinker`) | `deepseek` |
| `DEVBOT_MODEL` | Model id | `deepseek-v4-flash` |
| `LOCAL_LLM_BASE_URL` | Local OpenAI-compatible base URL | `http://127.0.0.1:8092/v1` |
| `LOCAL_LLM_MODEL` | Local model id | `vibethinker-q4-vulkan` |
| `LOCAL_LLM_API_KEY` | Local server API key placeholder | `local` |
| `LOCAL_LLM_MAX_TOKENS` | Max output tokens for local model calls | 600 |
| `DEVBOT_MAX_TURNS` | Max tool iterations per message | 200 |
| `DEVBOT_MAX_PARALLEL` | Max parallel agents in megaswarm | 8 |
| `DEVBOT_TOKEN_BUDGET` | Per-agent token cap (0 = unlimited) | 0 |
| `DEVBOT_GLOBAL_BUDGET` | Process-wide token cap (0 = unlimited) | 0 |
| `DEVBOT_COMPRESS_MODEL` | Model used for context compression | `deepseek-v4-flash` |
| `DEVBOT_VERBOSITY` | Set to `concise`, `terse`, `compact`, or `caveman` for shorter agent replies | normal |
| `DEVBOT_MAX_TOOL_OUTPUT` | Max chars returned to the model per tool call | 50000 |
| `DEVBOT_READ_FILE_LIMIT` | Default line count for `read_file` when no limit is passed | 2000 |
| `DEVBOT_DIFF_CLIP` | Max chars of edit diff returned to the model | 2000 |
| `DEVBOT_SPECIALIST_RESULT_LIMIT` | Max chars returned from one specialist/pipeline handoff | 8000 |
| `DEVBOT_MEGA_WARN_THRESHOLD` | Warn when N > threshold in megadelegate | 5 |
| `DEVBOT_PIPELINE_ROUNDS` | Max review→fix rounds in pipeline | 2 |
| `DEVBOT_SHOW_REASONING` | Show chain-of-thought (`1`, `true`, `yes`) | off |
| `DEVBOT_ALLOW_SHELL` | Skip shell approval for allow-listed commands (`1`, `true`) | off |
| `DEVBOT_LOG` | Path for JSONL structured log | off (no logging) |
| `DEVBOT_LOOP_LIMIT` | Consecutive identical tool calls / errors before halting (0 = off) | 3 |
| `DEVBOT_EVOLVE_MAX_PHASES` | Max phases per auto-evolve run | 20 |
| `DEVBOT_EVOLVE_TIME_LIMIT` | Max minutes per auto-evolve run (0 = unlimited) | 0 |

You can also put these in a `.env` file in your project root instead of exporting them:

```
DEEPSEEK_API_KEY=sk-...
DEVBOT_MODEL=deepseek-v4-flash
# For local VibeThinker instead:
# DEVBOT_PROVIDER=local-vibethinker
# LOCAL_LLM_BASE_URL=http://127.0.0.1:8092/v1
# LOCAL_LLM_MODEL=vibethinker-q4-vulkan
# LOCAL_LLM_API_KEY=local
# LOCAL_LLM_MAX_TOKENS=600
DEVBOT_VERBOSITY=concise
DEVBOT_MAX_TOOL_OUTPUT=12000
DEVBOT_SPECIALIST_RESULT_LIMIT=4000
```

Real environment variables take precedence over `.env`. Add `.env` to your `.gitignore`.

You can also put these settings in a `.devbot/config.toml` file in your project root:

```toml
# .devbot/config.toml (all keys optional)
model = "deepseek-v4-pro"
# Or use local VibeThinker:
# provider = "local-vibethinker"
# local_llm_base_url = "http://127.0.0.1:8092/v1"
# local_llm_model = "vibethinker-q4-vulkan"
# local_llm_api_key = "local"
# local_llm_max_tokens = 600
max_parallel = 4
token_budget = 100000
loop_limit = 5
verbosity = "concise"
max_tool_output = 12000
specialist_result_limit = 4000
```

Precedence: **environment variables** > `.env` file > `.devbot/config.toml` > defaults.
Missing or malformed config files are silently ignored.

## Safety model

DevBot is designed to be safe by default:

- **Sandboxed paths** — tools cannot read or write outside the project root. All paths are resolved and checked; `..` escapes and symlink tricks are rejected.
- **Shell block-list** — dangerous patterns (`rm -rf /`, `sudo`, `curl|sh`, fork bombs, etc.) are blocked outright. The model is told the command was rejected so it can adjust.
- **Shell allow-list** — safe/read-only commands (git status, pytest, ls, echo, cargo test, go vet, etc.) can be auto-approved with `DEVBOT_ALLOW_SHELL`.
- **Approval gates** — by default, writes, shell commands, and test runs require interactive confirmation (`y` / `a`lways / decline). The model sees the rejection so it can recover.
- **Mandatory review pipeline** — any code change (write_file, edit_file, or mutating run_command) MUST go through `pipeline(task)`, which enforces a coder→reviewer→fix loop. Direct writes by the manager are prohibited by system prompt.

In thinking mode the model's chain-of-thought is streamed (dimmed) before its answer.
