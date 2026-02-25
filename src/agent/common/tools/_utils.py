from __future__ import annotations


def truncate_text(value: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(value)
    if len(value) <= max_chars:
        return value, False
    return value[:max_chars], True
