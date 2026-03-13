"""Session-related types."""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

from agent.core.utils import to_non_negative_int


ContextBudgetSource = Literal["estimated", "provider"]


class ContextBudget(BaseModel):
    """上下文窗口预算快照"""

    total_tokens: int | None = None
    used_tokens: int = 0
    remaining_tokens: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    source: ContextBudgetSource = "estimated"
    updated_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    def update(
        self,
        *,
        total_tokens: int | None,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        source: ContextBudgetSource,
        updated_at_ms: int | None = None,
    ) -> None:
        """Update the context budget **in place**.

        Mutates ``self`` rather than returning a new instance so that all holders
        of a reference to this ``ContextBudget`` observe the change immediately.

        ``updated_at_ms`` is stored as epoch-milliseconds (not seconds) to match
        the precision used by provider usage events and to avoid floating-point
        rounding in JavaScript consumers.
        """
        self.total_tokens = to_non_negative_int(total_tokens) if total_tokens is not None else None
        self.input_tokens = to_non_negative_int(input_tokens)
        self.output_tokens = to_non_negative_int(output_tokens)
        self.reasoning_tokens = to_non_negative_int(reasoning_tokens)
        self.used_tokens = self.input_tokens + self.output_tokens + self.reasoning_tokens
        if self.total_tokens is None:
            self.remaining_tokens = None
        else:
            self.remaining_tokens = max(self.total_tokens - self.used_tokens, 0)
        self.source = "provider" if source == "provider" else "estimated"
        self.updated_at_ms = updated_at_ms if updated_at_ms is not None else int(time.time() * 1000)


class SessionMetadataData(BaseModel):
    session_id: str
    state: str
    model_profile_id: str
    agent_name: str
    context_budget: ContextBudget
    last_compaction_message_id: str | None = None
    created_at: str
    updated_at: str


class SessionData(BaseModel):
    session_id: str
    state: str
    model_profile_id: str
    agent_name: str
    context_budget: ContextBudget
    message_count: int
    messages: list[object]
    created_at: str
    updated_at: str
