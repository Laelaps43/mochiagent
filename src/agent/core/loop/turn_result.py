"""Typed turn result for LLM loop."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.core.compression import CompactionPayload
from agent.types import ContextBudget, TokenUsage, ToolCallPayload


class LLMTurnResult(BaseModel):
    content: str
    thinking: str
    tool_calls: list[ToolCallPayload] = Field(default_factory=list)
    finish_reason: str | None = None
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    context_budget: ContextBudget = Field(default_factory=ContextBudget)
    context_compaction: CompactionPayload = Field(
        default_factory=lambda: CompactionPayload.noop(stage="")
    )
    context_compaction_events: list[CompactionPayload] = Field(default_factory=list)
    message_id: str = ""
