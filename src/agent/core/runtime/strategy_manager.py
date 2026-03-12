"""Agent-level strategy facade for pluggable runtime behaviors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

from .context_compaction_manager import ContextCompactionManager
from .strategy_kind import StrategyKind
from .tool_postprocess_manager import ToolPostprocessManager

from agent.core.compression import CompactorFactory
from agent.core.tools import ToolResultPostProcessorFactory

if TYPE_CHECKING:
    from agent.core.compression import CompactionStage
    from agent.core.llm.base import LLMProvider
    from agent.core.session.context import SessionContext
    from agent.core.storage import StorageProvider
    from agent.types import ContextBudget, LLMConfig, ToolResult


class AgentStrategyManager:
    def __init__(self) -> None:
        self._compaction: ContextCompactionManager = ContextCompactionManager()
        self._postprocess: ToolPostprocessManager = ToolPostprocessManager()

    def register_compaction(self, name: str, factory: CompactorFactory) -> None:
        self._compaction.register(name, factory)

    def register_postprocess(self, name: str, factory: ToolResultPostProcessorFactory) -> None:
        self._postprocess.register(name, factory)

    def register(self, kind: StrategyKind, name: str, factory: Callable[..., object]) -> None:
        if kind == StrategyKind.CONTEXT_COMPACTION:
            self._compaction.register(name, cast(CompactorFactory, factory))
        else:
            self._postprocess.register(name, cast(ToolResultPostProcessorFactory, factory))

    def list(self, kind: StrategyKind) -> list[str]:
        if kind == StrategyKind.CONTEXT_COMPACTION:
            return self._compaction.list()
        return self._postprocess.list()

    def set_agent(
        self,
        kind: StrategyKind,
        agent_name: str,
        name: str,
        options: Mapping[str, object] | None = None,
    ) -> None:
        if kind == StrategyKind.CONTEXT_COMPACTION:
            self._compaction.set_agent(agent_name, name, options)
        else:
            self._postprocess.set_agent(agent_name, name, options)

    async def run_compaction(
        self,
        *,
        session_context: SessionContext,
        budget: ContextBudget,
        llm_config: LLMConfig,
        llm_provider: LLMProvider,
        agent_name: str | None = None,
        stage: CompactionStage,
        error: str | None = None,
    ) -> object:
        return await self._compaction.run(
            session_context=session_context,
            budget=budget,
            llm_config=llm_config,
            llm_provider=llm_provider,
            agent_name=agent_name,
            stage=stage,
            error=error,
        )

    async def run_postprocess(
        self,
        *,
        agent_name: str | None = None,
        session_id: str,
        tool_result: ToolResult,
        tool_arguments: Mapping[str, object],
        storage: StorageProvider,
    ) -> object:
        return await self._postprocess.run(
            agent_name=agent_name,
            session_id=session_id,
            tool_result=tool_result,
            tool_arguments=tool_arguments,
            storage=storage,
        )

    async def run(
        self,
        kind: StrategyKind,
        **kwargs: object,
    ) -> object:
        if kind == StrategyKind.CONTEXT_COMPACTION:
            return await self.run_compaction(
                session_context=cast("SessionContext", kwargs["session_context"]),
                budget=cast("ContextBudget", kwargs["budget"]),
                llm_config=cast("LLMConfig", kwargs["llm_config"]),
                llm_provider=cast("LLMProvider", kwargs["llm_provider"]),
                agent_name=cast("str | None", kwargs.get("agent_name")),
                stage=cast("CompactionStage", kwargs["stage"]),
                error=cast("str | None", kwargs.get("error")),
            )
        return await self.run_postprocess(
            agent_name=cast("str | None", kwargs.get("agent_name")),
            session_id=cast("str", kwargs["session_id"]),
            tool_result=cast("ToolResult", kwargs["tool_result"]),
            tool_arguments=cast("Mapping[str, object]", kwargs["tool_arguments"]),
            storage=cast("StorageProvider", kwargs["storage"]),
        )
