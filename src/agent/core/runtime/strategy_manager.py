"""Agent-level strategy facade for pluggable runtime behaviors."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .context_compaction_manager import ContextCompactionManager
from .strategy_kind import StrategyKind
from .tool_postprocess_manager import ToolPostprocessManager


class AgentStrategyManager:
    def __init__(self) -> None:
        self._managers: dict[StrategyKind, Any] = {
            StrategyKind.CONTEXT_COMPACTION: ContextCompactionManager(),
            StrategyKind.TOOL_RESULT_POSTPROCESS: ToolPostprocessManager(),
        }

    def register(self, kind: StrategyKind, name: str, factory: Callable[..., Any]) -> None:
        self._managers[kind].register(name, factory)

    def list(self, kind: StrategyKind) -> list[str]:
        return self._managers[kind].list()

    def set_agent(
        self,
        kind: StrategyKind,
        agent_name: str,
        name: str,
        options: Mapping[str, object] | None = None,
    ) -> None:
        self._managers[kind].set_agent(agent_name, name, options)

    async def run(
        self,
        kind: StrategyKind,
        **kwargs: Any,
    ) -> Any:
        return await self._managers[kind].run(**kwargs)
