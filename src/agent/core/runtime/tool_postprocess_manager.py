"""Tool result postprocess strategy manager."""

from __future__ import annotations

from typing import Any, Mapping

from loguru import logger

from agent.core.tools import (
    ToolPostprocessorConfig,
    ToolResultPostProcessor,
    ToolResultPostProcessConfig,
    ToolResultPostProcessorRegistry,
    ToolResultPostProcessorFactory,
)


class _AgentPostprocessorBinding:
    __slots__ = ("name", "config", "processor")

    def __init__(self, name: str, config: ToolPostprocessorConfig, processor: Any) -> None:
        self.name = name
        self.config = config
        self.processor = processor


class ToolPostprocessManager:
    def __init__(self) -> None:
        self._registry = ToolResultPostProcessorRegistry()
        self._registry.register("default", self._build_default_processor)
        self._default_name = "default"
        self._default = self._registry.create("default")
        self._agent_processors: dict[str, _AgentPostprocessorBinding] = {}

    @staticmethod
    def _build_default_processor(options: ToolPostprocessorConfig) -> ToolResultPostProcessor:
        config = ToolResultPostProcessConfig(
            summary_max_chars=int(options.get("summary_max_chars", 4000)),
            preview_head_chars=int(options.get("preview_head_chars", 1500)),
            preview_tail_chars=int(options.get("preview_tail_chars", 1000)),
        )
        return ToolResultPostProcessor(config)

    def register(self, name: str, factory: ToolResultPostProcessorFactory) -> None:
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
        resolved_config = ToolPostprocessorConfig.from_mapping(options)
        processor = self._registry.create(resolved_name, resolved_config)
        self._agent_processors[normalized] = _AgentPostprocessorBinding(
            name=resolved_name,
            config=resolved_config,
            processor=processor,
        )

    async def run(
        self,
        *,
        agent_name: str | None = None,
        session_id: str,
        tool_result: Any,
        tool_arguments: Mapping[str, object],
        storage: Any,
    ) -> Any:
        if agent_name and agent_name in self._agent_processors:
            binding = self._agent_processors[agent_name]
            processor_name, processor = binding.name, binding.processor
        else:
            processor_name, processor = (
                self._default_name,
                self._default,
            )
        try:
            return await processor.process(
                session_id=session_id,
                tool_result=tool_result,
                tool_arguments=tool_arguments,
                storage=storage,
            )
        except Exception as exc:
            logger.exception(
                "Tool result postprocessor '{}' failed: {}",
                processor_name,
                exc,
            )
            return tool_result
