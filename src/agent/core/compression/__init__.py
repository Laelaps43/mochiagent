"""Context compaction module."""

from .compactor import ContextCompactor, NoopContextCompactor
from .types import (
    CompactorRunOptions,
    CompactionDecision,
    CompactionPayload,
    CompactionResult,
    RewriteStats,
    StrategyConfig,
    SummaryBuildResult,
)
from .default_compactor import DefaultContextCompactor
from .registry import ContextCompactorRegistry, CompactorFactory
from .stage import CompactionStage

__all__ = [
    "CompactionResult",
    "CompactionDecision",
    "SummaryBuildResult",
    "RewriteStats",
    "CompactionPayload",
    "CompactorRunOptions",
    "StrategyConfig",
    "ContextCompactor",
    "NoopContextCompactor",
    "DefaultContextCompactor",
    "ContextCompactorRegistry",
    "CompactorFactory",
    "CompactionStage",
]
