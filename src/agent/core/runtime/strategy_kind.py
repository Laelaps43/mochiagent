"""Runtime strategy kinds."""

from __future__ import annotations

from enum import Enum


class StrategyKind(str, Enum):
    CONTEXT_COMPACTION = "context_compaction"
    TOOL_RESULT_POSTPROCESS = "tool_result_postprocess"
