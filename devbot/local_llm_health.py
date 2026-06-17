"""Health check for the local VibeThinker OpenAI-compatible server."""

from __future__ import annotations

import argparse
import os
import sys

import httpx

from .agent import (
    LOCAL_LLM_DEFAULT_API_KEY,
    LOCAL_LLM_DEFAULT_BASE_URL,
    LOCAL_LLM_DEFAULT_MODEL,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="devbot-local-health",
        description="Send a small chat-completions request to the local LLM server.",
    )
    parser.add_argument("--base-url", default=os.environ.get(
        "LOCAL_LLM_BASE_URL", LOCAL_LLM_DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get(
        "LOCAL_LLM_MODEL", LOCAL_LLM_DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.environ.get(
        "LOCAL_LLM_API_KEY", LOCAL_LLM_DEFAULT_API_KEY))
    parser.add_argument("--prompt", default="Hello")
    args = parser.parse_args()

    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": 64,
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {args.api_key}"}

    try:
        with httpx.Client(timeout=httpx.Timeout(60.0, connect=5.0)) as client:
            response = client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        print(f"[local-llm] health check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    message = data.get("choices", [{}])[0].get("message", {})
    content = (message.get("content") or "").strip()
    print(f"[local-llm] ok: {args.model} at {args.base_url}")
    if content:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
