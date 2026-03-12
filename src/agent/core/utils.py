from __future__ import annotations

from typing import TYPE_CHECKING

from agent.core._numeric import to_int, to_non_negative_int

if TYPE_CHECKING:
    from agent.types import ContextBudgetSource, ProviderUsage, TokenUsage

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


def make_token_usage(
    input: int = 0,
    output: int = 0,
    reasoning: int = 0,
) -> TokenUsage:
    from agent.types import TokenUsage

    return TokenUsage(
        input=max(input, 0),
        output=max(output, 0),
        reasoning=max(reasoning, 0),
    )


def extract_turn_tokens(usage: ProviderUsage | None) -> tuple[TokenUsage, ContextBudgetSource]:
    if usage is None:
        return make_token_usage(), "estimated"

    return make_token_usage(
        usage.input_tokens,
        usage.output_tokens,
        usage.reasoning_tokens,
    ), "provider"


def estimate_tokens(text_or_chars: str | int, chars_per_token: float) -> int:
    char_count = len(text_or_chars) if isinstance(text_or_chars, str) else text_or_chars
    return max(int(char_count / max(chars_per_token, 1.0)), 0)


def truncate_text(value: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(value)
    if len(value) <= max_chars:
        return value, False
    return value[:max_chars], True


def parse_name_list(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {item.strip().lower() for item in raw.split(",") if item and item.strip()}


def normalize_profile_id(profile_id: str) -> str:
    raw = profile_id.strip()
    if ":" not in raw:
        raise ValueError(
            f"Invalid model profile id '{profile_id}'. Expected format: provider:model"
        )
    provider, model = raw.split(":", 1)
    if not provider.strip() or not model.strip():
        raise ValueError("provider and model are required to build llm profile id")
    return f"{provider.strip().lower()}:{model.strip()}"
