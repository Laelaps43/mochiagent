"""Shared utility functions used across the agent core.

This module has **no** top-level imports from ``agent.types`` so that
``agent.types`` can safely import from here without circular dependency.
Functions that need runtime types use deferred imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.types import ProviderUsage, TokenUsage

__all__ = [
    "to_non_negative_int",
    "to_int",
    "make_token_usage",
    "extract_turn_tokens",
    "estimate_tokens",
    "truncate_text",
    "parse_name_list",
    "normalize_profile_id",
]


# ---------------------------------------------------------------------------
# Numeric conversions
# ---------------------------------------------------------------------------

def to_non_negative_int(value: object, *, default: int = 0) -> int:
    """Convert *value* to a non-negative ``int``.

    - ``bool`` values are treated as *default* (avoids ``True → 1``).
    - ``float`` values are truncated via ``int()``.
    - Unparseable values fall back to *default*.
    """
    if isinstance(value, bool):
        return default
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return default


def to_int(value: object, *, default: int = 0, minimum: int = 0) -> int:
    """Convert *value* to ``int``, clamped to *minimum*."""
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


# ---------------------------------------------------------------------------
# Token helpers  (deferred imports to avoid circular dep with agent.types)
# ---------------------------------------------------------------------------

def make_token_usage(
    input: int = 0,
    output: int = 0,
    reasoning: int = 0,
) -> TokenUsage:
    """Create a ``TokenUsage`` with guaranteed non-negative values."""
    from agent.types import TokenUsage

    return TokenUsage(
        input=max(input, 0),
        output=max(output, 0),
        reasoning=max(reasoning, 0),
    )


def extract_turn_tokens(usage: ProviderUsage | None) -> tuple[TokenUsage, str]:
    """Parse provider usage into ``(TokenUsage, source)``."""
    if usage is None:
        return make_token_usage(), "estimated"

    return make_token_usage(
        usage.input_tokens,
        usage.output_tokens,
        usage.reasoning_tokens,
    ), "provider"


def estimate_tokens(text_or_chars: str | int, chars_per_token: float) -> int:
    """Estimate token count from text (``str``) or character count (``int``)."""
    char_count = len(text_or_chars) if isinstance(text_or_chars, str) else text_or_chars
    return max(int(char_count / max(chars_per_token, 1.0)), 0)


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

def truncate_text(value: str, max_chars: int) -> tuple[str, bool]:
    """Truncate *value* to *max_chars*. Returns ``(text, was_truncated)``."""
    if max_chars <= 0:
        return "", bool(value)
    if len(value) <= max_chars:
        return value, False
    return value[:max_chars], True


def parse_name_list(raw: str | None) -> set[str]:
    """Parse a comma-separated string into a lowercase name set."""
    if not raw:
        return set()
    return {item.strip().lower() for item in raw.split(",") if item and item.strip()}


def normalize_profile_id(profile_id: str) -> str:
    """Normalize a ``provider:model`` profile id."""
    raw = profile_id.strip()
    if ":" not in raw:
        raise ValueError(
            f"Invalid model profile id '{profile_id}'. Expected format: provider:model"
        )
    provider, model = raw.split(":", 1)
    if not provider.strip() or not model.strip():
        raise ValueError("provider and model are required to build llm profile id")
    return f"{provider.strip().lower()}:{model.strip()}"
