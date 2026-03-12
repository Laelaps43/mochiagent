"""Type definitions for tool result postprocessor registry."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, Field


class ToolPostprocessorConfig(BaseModel):
    values: dict[str, object] = Field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None = None) -> ToolPostprocessorConfig:
        return cls(values=dict(values or {}))

    def get(self, key: str, default: object | None = None) -> object | None:
        return self.values.get(key, default)

    def __getitem__(self, key: str) -> object:
        return self.values[key]

    def __len__(self) -> int:
        return len(self.values)
