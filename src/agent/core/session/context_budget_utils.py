"""Context budget helpers."""

from __future__ import annotations

import time

from agent.types import ContextBudget, ContextBudgetData, ContextBudgetSource


def to_non_negative_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return default


def update_context_budget_from_raw(
    budget: ContextBudget,
    raw: ContextBudgetData | None,
) -> ContextBudget:
    if raw is None:
        return budget

    budget.total_tokens = to_non_negative_int(raw.total_tokens) if raw.total_tokens is not None else None
    budget.input_tokens = to_non_negative_int(raw.input_tokens)
    budget.output_tokens = to_non_negative_int(raw.output_tokens)
    budget.reasoning_tokens = to_non_negative_int(raw.reasoning_tokens)
    budget.used_tokens = to_non_negative_int(raw.used_tokens)

    if budget.total_tokens is None:
        budget.remaining_tokens = (
            to_non_negative_int(raw.remaining_tokens) if raw.remaining_tokens is not None else None
        )
    else:
        budget.remaining_tokens = max(budget.total_tokens - budget.used_tokens, 0)

    budget.source = raw.source
    budget.updated_at_ms = to_non_negative_int(raw.updated_at_ms, default=int(time.time() * 1000))
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
