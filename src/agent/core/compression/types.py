"""Type definitions for context compaction."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, Field


class CompactionResult(BaseModel):
    """Unified compaction output contract."""

    applied: bool
    reason: str = ""
    metadata: dict[str, object] = Field(default_factory=dict)
    stats: dict[str, object] = Field(default_factory=dict)
    artifacts: list[dict[str, object]] = Field(default_factory=list)


class StrategyConfig(BaseModel):
    """Typed wrapper for strategy-level configuration."""

    values: dict[str, object] = Field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None = None) -> StrategyConfig:
        return cls(values=dict(values or {}))

    def get(self, key: str, default: object | None = None) -> object | None:
        return self.values.get(key, default)

    def __getitem__(self, key: str) -> object:
        return self.values[key]

    def __len__(self) -> int:
        return len(self.values)


class CompactorRunOptions(BaseModel):
    """Typed runtime options for one compaction execution.

    策略配置项（创建 compactor 时确定）:
        auto_compact_ratio, keep_user_tokens_budget, chars_per_token,
        model_auto_compact_token_limit, summarization_prompt,
        summary_max_retries, summary_max_trims, summary_retry_sleep_ms.

    运行时信号（每次调用可能不同）:
        token_limit_reached, needs_follow_up.
    """

    # 策略配置
    auto_compact_ratio: float = 0.9
    keep_user_tokens_budget: int = 20000
    chars_per_token: float = 4.0
    model_auto_compact_token_limit: int | None = None
    summarization_prompt: str = ""
    summary_max_retries: int = 2
    summary_max_trims: int = 20
    summary_retry_sleep_ms: int = 300

    # 运行时信号
    token_limit_reached: bool = False
    needs_follow_up: bool = False

    @classmethod
    def from_config(cls, config: StrategyConfig) -> CompactorRunOptions:
        values = config.values
        field_names = {
            "auto_compact_ratio",
            "keep_user_tokens_budget",
            "chars_per_token",
            "model_auto_compact_token_limit",
            "summarization_prompt",
            "summary_max_retries",
            "summary_max_trims",
            "summary_retry_sleep_ms",
            "token_limit_reached",
            "needs_follow_up",
        }
        data = {f: values[f] for f in field_names if f in values}
        return cls.model_validate(data)


class CompactionPayload(BaseModel):
    """Normalized compaction payload for loop/runtime layers."""

    applied: bool
    reason: str
    metadata: dict[str, object] = Field(default_factory=dict)
    stats: dict[str, object] = Field(default_factory=dict)
    artifacts: list[dict[str, object]] = Field(default_factory=list)
    name: str = "unknown"
    stage: str = ""

    @classmethod
    def from_result(cls, result: CompactionResult, *, name: str, stage: str) -> CompactionPayload:
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
    ) -> CompactionPayload:
        return cls(
            applied=False,
            reason=reason,
            metadata={},
            stats={},
            artifacts=[],
            name=name,
            stage=stage,
        )


class CompactionDecision(BaseModel):
    """Trigger decision for a compaction attempt."""

    apply: bool
    reason: str
    metadata: dict[str, object] = Field(default_factory=dict)


class SummaryBuildResult(BaseModel):
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
    ) -> SummaryBuildResult:
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
    ) -> SummaryBuildResult:
        return cls(
            ok=False,
            error=error,
            trimmed_count=trimmed_count,
            retries=retries,
        )


class RewriteStats(BaseModel):
    """Context rewrite statistics."""

    before_messages: int
    after_messages: int
    retained_user_messages: int
    truncated_messages: int

    @property
    def dropped_messages(self) -> int:
        return self.before_messages - self.after_messages
