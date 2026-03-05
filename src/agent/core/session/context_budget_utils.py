"""Context budget helpers."""

from __future__ import annotations

import time

from agent.types import ContextBudget, ContextBudgetSource


def to_non_negative_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return default


def update_context_budget_from_raw(
    budget: ContextBudget,
    raw: object,
) -> ContextBudget:
    if not isinstance(raw, dict):
        return budget

    total_tokens = raw.get("total_tokens")
    budget.total_tokens = to_non_negative_int(total_tokens) if total_tokens is not None else None
    budget.input_tokens = to_non_negative_int(raw.get("input_tokens", 0))
    budget.output_tokens = to_non_negative_int(raw.get("output_tokens", 0))
    budget.reasoning_tokens = to_non_negative_int(raw.get("reasoning_tokens", 0))
    if "used_tokens" in raw:
        budget.used_tokens = to_non_negative_int(raw.get("used_tokens", 0))
    else:
        budget.used_tokens = budget.input_tokens + budget.output_tokens + budget.reasoning_tokens

    if budget.total_tokens is None:
        remaining_tokens = raw.get("remaining_tokens")
        budget.remaining_tokens = (
            to_non_negative_int(remaining_tokens) if remaining_tokens is not None else None
        )
    else:
        budget.remaining_tokens = max(budget.total_tokens - budget.used_tokens, 0)

    budget.source = "provider" if raw.get("source") == "provider" else "estimated"
    budget.updated_at_ms = to_non_negative_int(
        raw.get("updated_at_ms", int(time.time() * 1000)),
        default=int(time.time() * 1000),
    )
    return budget


def update_context_budget_values(
    budget: ContextBudget,
    *,
    total_tokens: int | None,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
    source: ContextBudgetSource,
    updated_at_ms: int | None = None,
) -> ContextBudget:
    budget.total_tokens = to_non_negative_int(total_tokens) if total_tokens is not None else None
    budget.input_tokens = to_non_negative_int(input_tokens)
    budget.output_tokens = to_non_negative_int(output_tokens)
    budget.reasoning_tokens = to_non_negative_int(reasoning_tokens)
    budget.used_tokens = budget.input_tokens + budget.output_tokens + budget.reasoning_tokens

    if budget.total_tokens is None:
        budget.remaining_tokens = None
    else:
        budget.remaining_tokens = max(budget.total_tokens - budget.used_tokens, 0)

    budget.source = "provider" if source == "provider" else "estimated"
    budget.updated_at_ms = updated_at_ms if updated_at_ms is not None else int(time.time() * 1000)
    return budget
