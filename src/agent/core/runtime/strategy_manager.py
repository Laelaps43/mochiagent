"""Agent-level strategy facade for pluggable runtime behaviors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Generic, TypeVar, cast

from loguru import logger

from .strategy_kind import StrategyKind

from agent.core.compression import (
    CompactionPayload,
    CompactionStage,
    CompactorRunOptions,
    ContextCompactor,
    DefaultContextCompactor,
)
from agent.core.tools import (
    ToolResultPostProcessor,
    ToolResultPostProcessorStrategy,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agent.core.llm.base import LLMProvider
    from agent.core.session.context import SessionContext
    from agent.core.storage import StorageProvider
    from agent.types import ContextBudget, Event, LLMConfig, ToolResult

S = TypeVar("S")


class StrategySlot(Generic[S]):
    """Per-agent strategy override with a shared default."""

    def __init__(self, default: S) -> None:
        self._default: S = default
        self._agents: dict[str, S] = {}

    def set(self, agent_name: str, strategy: S) -> None:
        self._agents[agent_name.strip()] = strategy

    def get(self, agent_name: str | None = None) -> S:
        if agent_name and agent_name in self._agents:
            return self._agents[agent_name]
        return self._default


class AgentStrategyManager:
    def __init__(self) -> None:
        self._compaction: StrategySlot[ContextCompactor] = StrategySlot(DefaultContextCompactor())
        self._postprocess: StrategySlot[ToolResultPostProcessorStrategy] = StrategySlot(
            ToolResultPostProcessor()
        )
        self._compaction_options: dict[str, CompactorRunOptions] = {}

    def set(
        self,
        kind: StrategyKind,
        agent_name: str,
        strategy: object,
        *,
        compaction_options: CompactorRunOptions | None = None,
    ) -> None:
        """Set a per-agent strategy instance by kind."""
        if kind == StrategyKind.CONTEXT_COMPACTION:
            self._compaction.set(agent_name, cast(ContextCompactor, strategy))
            if compaction_options:
                self._compaction_options[agent_name.strip()] = compaction_options
        elif kind == StrategyKind.TOOL_RESULT_POSTPROCESS:
            self._postprocess.set(agent_name, cast(ToolResultPostProcessorStrategy, strategy))

    async def run_compaction(
        self,
        *,
        session_context: SessionContext,
        budget: ContextBudget,
        llm_config: LLMConfig,
        llm_provider: LLMProvider,
        agent_name: str | None = None,
        stage: CompactionStage,
        error: str | None = None,  # noqa: ARG002
        emit_event: Callable[[Event], Awaitable[None]] | None = None,
    ) -> CompactionPayload:
        _ = error  # reserved for future per-stage error context
        compactor = self._compaction.get(agent_name)
        options = (
            self._compaction_options.get(agent_name) if agent_name else None
        ) or CompactorRunOptions()

        return await compactor.run(
            session_context=session_context,
            budget=budget,
            llm_config=llm_config,
            llm_provider=llm_provider,
            stage=stage,
            options=options,
            emit_event=emit_event,
        )

    async def run_postprocess(
        self,
        *,
        agent_name: str | None = None,
        session_id: str,
        tool_result: ToolResult,
        tool_arguments: Mapping[str, object],
        storage: StorageProvider,
    ) -> ToolResult:
        processor = self._postprocess.get(agent_name)
        try:
            return await processor.process(
                session_id=session_id,
                tool_result=tool_result,
                tool_arguments=tool_arguments,
                storage=storage,
            )
        except Exception as exc:
            logger.exception(
                "Tool result postprocessor failed: {}",
                exc,
            )
            return tool_result
