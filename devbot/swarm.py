"""Agent swarm: a manager agent that delegates tasks to specialist sub-agents.

Pattern: the manager gets a `delegate(role, task)` tool. Each call spawns a
fresh Agent with a specialist system prompt and a restricted tool set, runs its
own agentic loop, and returns the specialist's final answer to the manager.

Specialists never get the delegate tool themselves, so there is no recursion.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a circular import at runtime (agent.py imports nothing here)
    from .agent import Agent

# All read-only tools, used by review/research style specialists.
_READ_ONLY = ["read_file", "list_dir", "grep", "find_files", "web_search"]

# role -> config. `tools=None` means "all tools"; `model=None` inherits manager's.
SPECIALISTS: dict[str, dict] = {
    "coder": {
        "description": "Writes, edits, and refactors code. Full tool access.",
        "tools": None,
        "model": None,
        "prompt": (
            "You are a specialist CODER agent in a swarm. You receive a single, "
            "self-contained coding task from a manager agent. Inspect the relevant "
            "files, make the change, and verify it (run tests or the script) when "
            "possible. Return a concise summary of exactly what you changed and any "
            "follow-ups the manager should know about."
        ),
    },
    "reviewer": {
        "description": "Reviews code for bugs, edge cases, and style. Read-only.",
        "tools": _READ_ONLY,
        "model": None,
        "prompt": (
            "You are a specialist code REVIEWER in a swarm. You have read-only tools. "
            "Review the code you are asked about for correctness, edge cases, security, "
            "and readability. Be specific: cite file:line and give actionable fixes. "
            "Do not attempt to edit files."
        ),
    },
    "tester": {
        "description": "Runs tests, diagnoses failures, suggests fixes.",
        "tools": _READ_ONLY + ["run_command", "verify"],
        "model": None,
        "prompt": (
            "You are a specialist TESTER in a swarm. Run the project's tests, read the "
            "output, identify the root cause of any failures, and report specific fixes. "
            "You may run commands but should not edit files — leave fixes to the coder."
        ),
    },
    "researcher": {
        "description": "Explores the codebase and answers questions. Read-only.",
        "tools": _READ_ONLY,
        "model": None,
        "prompt": (
            "You are a specialist RESEARCHER in a swarm. Explore the codebase to answer "
            "the manager's question thoroughly. Read relevant files, search for patterns, "
            "and cite specific file:line locations in your answer."
        ),
    },
}


def delegate_schema() -> dict:
    """The `delegate` tool definition added to the manager's tool set."""
    roles = ", ".join(f"{r} ({c['description']})" for r, c in SPECIALISTS.items())
    return {
        "type": "function",
        "function": {
            "name": "delegate",
            "description": (
                "Delegate a self-contained subtask to a specialist sub-agent, which "
                "works independently and returns its final result. Available roles: "
                + roles
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": list(SPECIALISTS),
                        "description": "Which specialist to delegate to.",
                    },
                    "task": {
                        "type": "string",
                        "description": "A clear, self-contained description of the subtask.",
                    },
                },
                "required": ["role", "task"],
            },
        },
    }


def run_specialist(manager: "Agent", role: str, task: str) -> str:
    """Spawn a specialist sub-agent, run the task, return its final answer."""
    from .agent import Agent, TOOL_SCHEMAS  # local import avoids circular dependency

    spec = SPECIALISTS.get(role)
    if spec is None:
        return f"Error: unknown specialist '{role}'. Choose one of: {', '.join(SPECIALISTS)}"

    allowed = spec["tools"]
    schemas = (TOOL_SCHEMAS if allowed is None
               else [s for s in TOOL_SCHEMAS if s["function"]["name"] in allowed])

    sub = Agent(
        root=manager.root,
        model=spec["model"] or manager.model,
        auto_approve=manager.auto_approve,
        system_prompt=spec["prompt"],
        tool_schemas=schemas,
        label=role,
    )
    # Share the already-authenticated client so we don't re-read env / re-auth.
    sub.client = manager.client

    print(f"\n\x1b[35m╭─ delegating to [{role}]\x1b[0m \x1b[90m{task[:100]}\x1b[0m")
    start = time.time()
    answer = sub.run(task)
    elapsed = time.time() - start
    print(f"\n\x1b[35m╰─ [{role}] done in {elapsed:.1f}s\x1b[0m")

    # Roll the sub-agent's token usage up into the manager's session totals.
    manager.total_tokens += sub.total_tokens
    manager.delegation_count += 1
    return answer or "(specialist returned no text)"
