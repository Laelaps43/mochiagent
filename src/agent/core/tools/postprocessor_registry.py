"""Tool result postprocessor strategy registry."""

from __future__ import annotations

from typing import Callable, Mapping

from .postprocessor_types import ToolPostprocessorConfig
from .result_postprocessor import ToolResultPostProcessorStrategy

ToolResultPostProcessorFactory = Callable[[ToolPostprocessorConfig], ToolResultPostProcessorStrategy]


class ToolResultPostProcessorRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ToolResultPostProcessorFactory] = {}

    def register(self, name: str, factory: ToolResultPostProcessorFactory) -> None:
        if not name or not name.strip():
            raise ValueError("postprocessor name is required")
        self._factories[name.strip().lower()] = factory

    def create(
        self,
        name: str,
        options: Mapping[str, object] | ToolPostprocessorConfig | None = None,
    ) -> ToolResultPostProcessorStrategy:
        key = name.strip().lower()
        if key not in self._factories:
            available = ", ".join(sorted(self._factories.keys())) or "<none>"
            raise ValueError(f"Unknown tool result postprocessor '{name}'. Available: {available}")
        if isinstance(options, ToolPostprocessorConfig):
            config = options
        else:
            config = ToolPostprocessorConfig.from_mapping(options)
        return self._factories[key](config)

    def list(self) -> list[str]:
        return sorted(self._factories.keys())
