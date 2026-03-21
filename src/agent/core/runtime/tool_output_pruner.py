"""Tool output pruner — clears old tool outputs to free context space.

Walks backward through messages, counting user messages as "turns".
Tool outputs older than `max_turns` are replaced with a compacted placeholder.
"""

from __future__ import annotations

from collections.abc import Sequence

from loguru import logger
from pydantic import BaseModel

from agent.core.message import Message as InternalMessage
from agent.core.message.part import ToolPart


COMPACTED_PLACEHOLDER = "[compacted: tool output removed to free context]"


class ToolOutputPrunerConfig(BaseModel):
    max_turns: int = 25


class ToolOutputPruner:
    def __init__(self, config: ToolOutputPrunerConfig | None = None) -> None:
        self.config: ToolOutputPrunerConfig = config or ToolOutputPrunerConfig()

    def prune(self, messages: Sequence[InternalMessage]) -> int:
        """Prune old tool outputs in-place. Returns number of parts pruned."""
        max_turns = self.config.max_turns
        if max_turns <= 0:
            return 0

        # Find the cutoff index: walk backward, count user messages
        user_count = 0
        cutoff_idx = 0
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                user_count += 1
                if user_count > max_turns:
                    cutoff_idx = i + 1
                    break

        if cutoff_idx == 0:
            return 0  # Not enough turns to prune

        # Prune all tool outputs before the cutoff
        pruned = 0
        for msg in messages[:cutoff_idx]:
            for part in msg.parts:
                if not isinstance(part, ToolPart):
                    continue
                if part.state.status != "completed":
                    continue
                # Already pruned
                if part.state.output == COMPACTED_PLACEHOLDER:
                    continue

                part.state.output = COMPACTED_PLACEHOLDER
                part.state.summary = COMPACTED_PLACEHOLDER
                pruned += 1

        if pruned:
            logger.info(
                "Pruned {} tool output(s) older than {} turns",
                pruned,
                max_turns,
            )

        return pruned
