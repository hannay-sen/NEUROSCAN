"""Thin Claude / OpenAI wrapper. Picks a provider from the API keys in .env."""
from __future__ import annotations

import os

from neuroscan.config import PROJECT_ROOT, load_config

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


class LLMUnavailable(RuntimeError):
    pass


def provider() -> str:
    want = load_config()["llm"]["provider"]
    if want in ("anthropic", "openai", "none"):
        return want
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "none"


def available() -> bool:
    return provider() != "none"


def complete(prompt: str, system: str = "", max_tokens: int | None = None) -> str:
    cfg = load_config()["llm"]
    max_tokens = max_tokens or cfg["max_tokens"]
    p = provider()
    if p == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=cfg["anthropic_model"], max_tokens=max_tokens, temperature=cfg["temperature"],
            system=system or anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    if p == "openai":
        from openai import OpenAI

        client = OpenAI()
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        r = client.chat.completions.create(model=cfg["openai_model"], messages=messages,
                                           max_tokens=max_tokens, temperature=cfg["temperature"])
        return (r.choices[0].message.content or "").strip()
    raise LLMUnavailable("No LLM configured: add ANTHROPIC_API_KEY or OPENAI_API_KEY to .env")
