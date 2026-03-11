"""Context compaction interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .stage import CompactionStage
from .types import CompactorRunOptions, CompactionResult

if TYPE_CHECKING:
    from agent.core.session import SessionContext
    from agent.core.llm import LLMProvider
    from agent.types import ContextBudget, LLMConfig


class ContextCompactor(ABC):
    """Single extension point: decide and execute context compaction."""

    @abstractmethod
    async def run(
        self,
        *,
        session_context: "SessionContext",
        budget: "ContextBudget",
        llm_config: "LLMConfig",
        llm_provider: "LLMProvider",
        stage: CompactionStage,
        error: str | None = None,
        options: CompactorRunOptions,
    ) -> CompactionResult:
        """
        Run compaction logic.

        The implementation can decide whether to compact (trigger) and how to compact.
        It may mutate ``session_context`` directly.
        """
        raise NotImplementedError


class NoopContextCompactor(ContextCompactor):
    """Default compactor that never compacts."""

    async def run(
        self,
        *,
        session_context: "SessionContext",
        budget: "ContextBudget",
        llm_config: "LLMConfig",
        llm_provider: "LLMProvider",
        stage: CompactionStage,
        error: str | None = None,
        options: CompactorRunOptions,
    ) -> CompactionResult:
        return CompactionResult(applied=False, reason="noop")
