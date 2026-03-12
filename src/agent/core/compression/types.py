"""Type definitions for context compaction."""

from __future__ import annotations

from pydantic import BaseModel, Field


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


class CompactionPayload(BaseModel):
    """Compaction result — the bookmark is already in the context, this is just a signal."""

    applied: bool
    reason: str = ""
    name: str = ""
    stage: str = ""

    @classmethod
    def noop(cls, *, stage: str, reason: str = "noop") -> CompactionPayload:
        return cls(applied=False, reason=reason, stage=stage)


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
        trimmed_count: int = 0,
        retries: int = 0,
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
        trimmed_count: int = 0,
        retries: int = 0,
    ) -> SummaryBuildResult:
        return cls(
            ok=False,
            error=error,
            trimmed_count=trimmed_count,
            retries=retries,
        )
