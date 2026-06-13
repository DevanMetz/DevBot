"""Live smoke test: run a trivial megaswarm task and verify the dashboard.

This is a MANUAL script, not a pytest unit test — it makes real (paid) API
calls. Guarded under __main__ so `pytest` collects it without executing
anything (which would otherwise fail CI / cost tokens on every test run).
Run it directly:  python tests/test_phase3_smoke.py
"""
import sys, time
from pathlib import Path

from devbot.agent import Agent


def main():
    root = Path(__file__).resolve().parent.parent
    agent = Agent(
        root=root,
        model="deepseek-v4-flash",
        auto_approve=True,   # -y: no interactive prompts
        megaswarm=True,
    )

    print("=== Live Phase 3 smoke test ===")
    print(f"TTY: {sys.stdout.isatty()}")
    print(f"Model: {agent.model}")
    print()

    start = time.time()
    result = agent.run("Read plan.md and tell me in one sentence what Phase 3 is about.")
    elapsed = time.time() - start

    print(f"\n=== Done in {elapsed:.1f}s ===")
    print(f"Tokens: {agent.total_tokens:,}")
    print(f"Result length: {len(result)} chars")
    assert len(result) > 20, f"Result too short: {result[:100]}"
    print("PASS: got a non-trivial response")


if __name__ == "__main__":
    main()
