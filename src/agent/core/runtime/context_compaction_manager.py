"""Context compaction strategy manager."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from loguru import logger

from agent.core.compression import (
    CompactionPayload,
    CompactorRunOptions,
    CompactionResult,
    CompactionStage,
    ContextCompactor,
    ContextCompactorRegistry,
    CompactorFactory,
    DefaultContextCompactor,
    NoopContextCompactor,
    StrategyConfig,
)
from agent.types import ContextBudget, LLMConfig

if TYPE_CHECKING:
    from agent.core.llm.base import LLMProvider
    from agent.core.session.context import SessionContext


class _AgentCompactorBinding:
    __slots__ = ("name", "config", "compactor")

    def __init__(self, name: str, config: StrategyConfig, compactor: ContextCompactor) -> None:
        self.name = name
        self.config = config
        self.compactor = compactor


class ContextCompactionManager:
    def __init__(self) -> None:
        self._registry = ContextCompactorRegistry()
        self._registry.register("default", lambda _opts: DefaultContextCompactor())
        self._registry.register("noop", lambda _opts: NoopContextCompactor())
        self._default_name = "default"
        self._default = self._registry.create("default")
        self._agent_compactors: dict[str, _AgentCompactorBinding] = {}

    def register(self, name: str, factory: CompactorFactory) -> None:
        self._registry.register(name, factory)

    def list(self) -> list[str]:
        return self._registry.list()

    def set_agent(
        self,
        agent_name: str,
        name: str,
        options: Mapping[str, object] | None = None,
    ) -> None:
        normalized = agent_name.strip()
        if not normalized:
            raise ValueError("agent_name is required")
        resolved_name = name.strip().lower()
        resolved_config = StrategyConfig.from_mapping(options)
        compactor = self._registry.create(resolved_name, resolved_config)
        self._agent_compactors[normalized] = _AgentCompactorBinding(
            name=resolved_name,
            config=resolved_config,
            compactor=compactor,
        )

    async def run(
        self,
        *,
        session_context: SessionContext,
        budget: ContextBudget,
        llm_config: LLMConfig,
        llm_provider: LLMProvider,
        agent_name: str | None = None,
        stage: CompactionStage,
        error: str | None = None,
    ) -> CompactionPayload:
        stage_value = stage.value
        if agent_name and agent_name in self._agent_compactors:
            binding = self._agent_compactors[agent_name]
            compactor_name, compactor_config, compactor = (
                binding.name,
                binding.config,
                binding.compactor,
            )
        else:
            compactor_name, compactor_config, compactor = (
                self._default_name,
                StrategyConfig(),
                self._default,
            )
        try:
            result = await compactor.run(
                session_context=session_context,
                budget=budget,
                llm_config=llm_config,
                llm_provider=llm_provider,
                stage=stage,
                error=error,
                options=CompactorRunOptions.from_config(compactor_config),
            )
        except Exception as exc:
            logger.exception(
                "Context compactor '{}' failed at stage '{}': {}",
                compactor_name,
                stage_value,
                exc,
            )
            result = CompactionResult(
                applied=False,
                reason=f"compactor_error: {type(exc).__name__}",
                metadata={"error": str(exc)},
            )

        return CompactionPayload.from_result(
            result,
            name=compactor_name,
            stage=stage_value,
        )
