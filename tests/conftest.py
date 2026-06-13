"""Shared pytest setup.

Some tests construct a real ``Agent``, whose ``__init__`` raises ``SystemExit``
when ``DEEPSEEK_API_KEY`` is unset (e.g. in CI). Provide a dummy key so the
suite collects and runs without network access. A real key in the environment
still takes precedence (``setdefault``), and no test makes live API calls.
"""
import os

os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-dummy")
