# DevBot

A Claude Code–style CLI coding agent powered by the [DeepSeek API](https://platform.deepseek.com).

DevBot runs an agentic loop in your terminal: you describe a task, the model reads
your code with tools, edits files, and runs commands (with your approval) until the
task is done.

## Features

- **Streaming agentic loop** with OpenAI-compatible function calling against DeepSeek.
- **Six built-in tools** for reading, searching, editing, and running code.
- **Approval gates** — writes and shell commands require confirmation (`y` / `a`lways / decline).
- **Sandboxed file access** — tools cannot read or write outside the project root.
- **Reasoning support** — `deepseek-reasoner` chain-of-thought is streamed dimmed.
- **Resilient** — transient API errors are retried with exponential backoff; auth/balance
  errors give clear, actionable messages.
- **Token tracking** with a context-window warning, plus REPL history and `.env` support.

## Tools available to the model

| Tool | What it does | Needs approval |
|---|---|---|
| `read_file` | Read a file with line numbers | no |
| `list_dir` | List a directory | no |
| `grep` | Regex search across the project | no |
| `find_files` | Find files by name glob (e.g. `*.py`) | no |
| `write_file` | Create/overwrite a file | yes |
| `edit_file` | Exact string replacement in a file | yes |
| `run_command` | Shell command in the project root | yes |

## Setup

```sh
pip install -e .
```

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
devbot -m deepseek-reasoner  # use the reasoning model for harder tasks
devbot -C path/to/project    # operate on another directory
```

In the REPL:

- `/help` — list commands
- `/clear` — reset the conversation (keeps system prompt)
- `/stats` — show token usage (last prompt + session total) and message count
- `/model <name>` — switch model (`deepseek-chat` or `deepseek-reasoner`; warns on unknown names)
- `/auto` — toggle auto-approve
- `/exit` — quit

DevBot tracks token usage per call and warns when a prompt approaches the 64K
context limit (use `/clear` to reset). Set `DEVBOT_MAX_TURNS` to cap tool
iterations per message (default 40).

When the model wants to write a file or run a command you'll get a prompt:
`y` allows once, `a` allows everything for the rest of the session, anything
else declines (the model is told and can adjust).

## Configuration

- `DEEPSEEK_API_KEY` (required)
- `DEVBOT_MODEL` — default model, overridden by `-m`
- `DEVBOT_MAX_TURNS` — max tool iterations per message (default 40)

You can also put these in a `.env` file in your project root instead of exporting them:

```
DEEPSEEK_API_KEY=sk-...
DEVBOT_MODEL=deepseek-chat
```

Real environment variables take precedence over `.env`. Add `.env` to your `.gitignore`.

Using `-m deepseek-reasoner` streams the model's chain-of-thought (dimmed) before its answer.
