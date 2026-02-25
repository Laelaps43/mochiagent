"""Message normalization helpers for system prompt injection."""

from __future__ import annotations


def inject_system_prompt(
    messages: list[dict],
    system_prompt: str | None,
) -> list[dict]:
    """Return a normalized message list with at most one system message at index 0."""
    sanitized = [msg for msg in messages if msg.get("role") != "system"]
    content = (system_prompt or "").strip()
    if not content:
        return sanitized
    return [{"role": "system", "content": content}] + sanitized
