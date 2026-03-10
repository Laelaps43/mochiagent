"""Typed turn result for LLM loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.core.compression import CompactionPayload
from agent.types import ContextBudget, TokenUsage, ToolCallPayload


@dataclass(slots=True)
class LLMTurnResult:
    content: str
    thinking: str
    tool_calls: list[ToolCallPayload] = field(default_factory=list)
    finish_reason: str | None = None
    cost: float = 0.0
    tokens: TokenUsage = field(default_factory=lambda: {"input": 0, "output": 0, "reasoning": 0})
    context_budget: ContextBudget = field(default_factory=ContextBudget)
    context_compaction: CompactionPayload = field(
        default_factory=lambda: CompactionPayload.invalid(stage="")
    )
    context_compaction_events: list[CompactionPayload] = field(default_factory=list)
    message_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "thinking": self.thinking,
            "tool_calls": self.tool_calls,
            "finish_reason": self.finish_reason,
            "cost": self.cost,
            "tokens": self.tokens,
            "context_budget": self.context_budget,
            "context_compaction": self.context_compaction.to_dict(),
            "context_compaction_events": [item.to_dict() for item in self.context_compaction_events],
            "message_id": self.message_id,
        }
