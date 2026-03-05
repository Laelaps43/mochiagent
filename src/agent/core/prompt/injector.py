"""Message normalization helpers for system prompt injection."""

from __future__ import annotations

from agent.types import Message as ChatMessage, MessageRole


def inject_system_prompt(
    messages: list[ChatMessage],
    system_prompt: str | None,
) -> list[ChatMessage]:
    """Return a normalized message list with at most one system message at index 0."""
    sanitized = [msg for msg in messages if msg.role != MessageRole.SYSTEM]
    content = (system_prompt or "").strip()
    if not content:
        return sanitized
    return [ChatMessage(role=MessageRole.SYSTEM, content=content)] + sanitized
