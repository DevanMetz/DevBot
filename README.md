# DevBot

A Claude Code–style CLI coding agent powered by the [DeepSeek API](https://platform.deepseek.com).

DevBot runs an agentic loop in your terminal: you describe a task, the model reads
your code with tools, edits files, and runs commands (with your approval) until the
task is done.

## Features

- **Streaming agentic loop** with OpenAI-compatible function calling against DeepSeek.
- **Nine built-in tools** for reading, searching, editing, running, web-searching, and testing code.
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
| `grep` | Regex search across the project | no |
| `find_files` | Find files by name glob (e.g. `*.py`) | no |
| `web_search` | Web search via DuckDuckGo (titles, URLs, snippets) | no |
| `write_file` | Create/overwrite a file | yes |
| `edit_file` | Exact string replacement in a file | yes |
| `run_command` | Shell command in the project root | yes |
| `verify` | Auto-detect & run the project's test suite | yes |

## Setup

```sh
pip install -e .
```

This installs the `openai` SDK and `ddgs` (DuckDuckGo search, used by `web_search`).

Get an API key at https://platform.deepseek.com, then:

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."   # PowerShell
# export DEEPSEEK_API_KEY=sk-...   # bash/zsh
```

## Usage

```sh
devbot                       # interactive REPL in the current directory
devbot "fix the failing test"  # one-shot mode
devbot -y                    # auto-approve all tool calls (be careful)
devbot -m deepseek-v4-pro    # use the more capable model for harder tasks
devbot -C path/to/project    # operate on another directory
```

In the REPL:

- `/help` — list commands
- `/clear` — reset the conversation (keeps system prompt)
- `/stats` — show token usage (last prompt + session total) and message count
- `/model <name>` — switch model (`deepseek-v4-flash` or `deepseek-v4-pro`; warns on unknown names)
- `/auto` — toggle auto-approve
- `/swarm` — toggle swarm mode (resets the conversation)
- `/exit` — quit

DevBot tracks token usage per call and warns (and auto-compresses) when a prompt
approaches the 1M context limit. Set `DEVBOT_MAX_TURNS` to cap tool iterations
per message (default 40).

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

## Configuration

- `DEEPSEEK_API_KEY` (required)
- `DEVBOT_MODEL` — default model, overridden by `-m`
- `DEVBOT_MAX_TURNS` — max tool iterations per message (default 40)

You can also put these in a `.env` file in your project root instead of exporting them:

```
DEEPSEEK_API_KEY=sk-...
DEVBOT_MODEL=deepseek-v4-flash
```

Real environment variables take precedence over `.env`. Add `.env` to your `.gitignore`.

In thinking mode the model's chain-of-thought is streamed (dimmed) before its answer.
