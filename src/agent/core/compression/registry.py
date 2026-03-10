"""Context compactor registry."""

from __future__ import annotations

from typing import Callable, Mapping

from .compactor import ContextCompactor
from .types import StrategyConfig

CompactorFactory = Callable[[StrategyConfig], ContextCompactor]


class ContextCompactorRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, CompactorFactory] = {}

    def register(self, name: str, factory: CompactorFactory) -> None:
        if not name or not name.strip():
            raise ValueError("compactor name is required")
        key = name.strip().lower()
        self._factories[key] = factory

    def has(self, name: str) -> bool:
        return name.strip().lower() in self._factories

    def create(self, name: str, options: Mapping[str, object] | StrategyConfig | None = None) -> ContextCompactor:
        key = name.strip().lower()
        if key not in self._factories:
            available = ", ".join(sorted(self._factories.keys())) or "<none>"
            raise ValueError(f"Unknown context compactor '{name}'. Available: {available}")
        if isinstance(options, StrategyConfig):
            config = options
        else:
            config = StrategyConfig.from_mapping(options)
        return self._factories[key](config)

    def list(self) -> list[str]:
        return sorted(self._factories.keys())
