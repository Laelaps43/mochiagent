"""Type definitions for context compaction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping


@dataclass(slots=True)
class CompactionResult:
    """Unified compaction output contract."""

    applied: bool
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "reason": self.reason,
            "metadata": self.metadata,
            "stats": self.stats,
            "artifacts": self.artifacts,
        }


@dataclass(slots=True)
class StrategyConfig:
    """Typed wrapper for strategy-level configuration."""

    values: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None = None) -> "StrategyConfig":
        return cls(values=dict(values or {}))

    def get(self, key: str, default: object | None = None) -> object | None:
        return self.values.get(key, default)

    def __getitem__(self, key: str) -> object:
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


@dataclass(slots=True)
class CompactorRunOptions:
    """Typed runtime options for one compaction execution."""

    stage: str
    error: str | None = None
    values: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        *,
        config: StrategyConfig,
        stage: str,
        error: str | None = None,
    ) -> "CompactorRunOptions":
        return cls(stage=stage, error=error, values=dict(config.values))

    def get(self, key: str, default: object | None = None) -> object | None:
        if key == "stage":
            return self.stage
        if key == "error":
            return self.error
        return self.values.get(key, default)

    def __getitem__(self, key: str) -> object | None:
        value = self.get(key)
        if value is None and key not in {"error", "stage"} and key not in self.values:
            raise KeyError(key)
        return value


@dataclass(slots=True)
class CompactionPayload:
    """Normalized compaction payload for loop/runtime layers."""

    applied: bool
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    name: str = "unknown"
    stage: str = ""

    @classmethod
    def from_result(cls, result: CompactionResult, *, name: str, stage: str) -> "CompactionPayload":
        return cls(
            applied=result.applied,
            reason=result.reason,
            metadata=result.metadata,
            stats=result.stats,
            artifacts=result.artifacts,
            name=name,
            stage=stage,
        )

    @classmethod
    def invalid(
        cls,
        *,
        stage: str,
        reason: str = "invalid_compactor_result",
        name: str = "unknown",
    ) -> "CompactionPayload":
        return cls(
            applied=False,
            reason=reason,
            metadata={},
            stats={},
            artifacts=[],
            name=name,
            stage=stage,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "reason": self.reason,
            "metadata": self.metadata,
            "stats": self.stats,
            "artifacts": self.artifacts,
            "name": self.name,
            "stage": self.stage,
        }


@dataclass(slots=True)
class CompactionDecision:
    """Trigger decision for a compaction attempt."""

    apply: bool
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SummaryBuildResult:
    """Summary generation result."""

    ok: bool
    summary_text: str = ""
    error: str = ""
    trimmed_count: int = 0
    retries: int = 0

    @classmethod
    def success(
        cls,
        *,
        summary_text: str,
        trimmed_count: int,
        retries: int,
    ) -> "SummaryBuildResult":
        return cls(
            ok=True,
            summary_text=summary_text,
            trimmed_count=trimmed_count,
            retries=retries,
        )

    @classmethod
    def failure(
        cls,
        *,
        error: str,
        trimmed_count: int,
        retries: int,
    ) -> "SummaryBuildResult":
        return cls(
            ok=False,
            error=error,
            trimmed_count=trimmed_count,
            retries=retries,
        )


@dataclass(slots=True)
class RewriteStats:
    """Context rewrite statistics."""

    before_messages: int
    after_messages: int
    retained_user_messages: int
    truncated_messages: int

    @property
    def dropped_messages(self) -> int:
        return self.before_messages - self.after_messages
