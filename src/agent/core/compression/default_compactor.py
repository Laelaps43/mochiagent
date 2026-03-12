"""Default context compactor implementation."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, override

from agent.core.llm.errors import is_context_overflow_error as _is_context_overflow
from agent.types import LLMConfig, LLMStreamChunk

from agent.core.compression.compactor import ContextCompactor
from agent.core.compression.types import (
    CompactorRunOptions,
    CompactionDecision,
    SummaryBuildResult,
)
from agent.core.message import (
    Message as InternalMessage,
    TextPart,
    ToolPart,
    UserMessageInfo,
)
from agent.core.compression.stage import CompactionStage
from agent.core.utils import estimate_tokens

if TYPE_CHECKING:
    from agent.core.llm.base import LLMProvider
    from agent.types import ContextBudget


SUMMARIZATION_PROMPT = (
    "You are performing a CONTEXT CHECKPOINT COMPACTION.\n"
    "Create a handoff summary for another LLM that will resume the task.\n"
    "Include:\n"
    "- Current progress and key decisions made\n"
    "- Important context, constraints, or user preferences\n"
    "- What remains to be done (clear next steps)\n"
    "- Any critical data, examples, or references needed to continue\n"
    "Be concise, structured, and focused on helping the next LLM seamlessly continue the work."
)
SUMMARY_PREFIX = "COMPACTION_SUMMARY\n"


class DefaultContextCompactor(ContextCompactor):
    """Default built-in compaction strategy.

    Trigger points:
    - pre_call: automatic threshold-based compaction
    - overflow_error: force compact after provider context overflow
    - mid_turn: compact only when token limit was reached and follow-up is required
    """

    @override
    def should_compact(
        self,
        *,
        messages: Sequence[InternalMessage],
        budget: ContextBudget,
        llm_config: LLMConfig,
        stage: CompactionStage,
        options: CompactorRunOptions,
    ) -> CompactionDecision:
        """决定"是否压缩"以及"触发原因"。"""
        stage_value = stage.value

        if stage is CompactionStage.OVERFLOW_ERROR:
            return CompactionDecision(
                apply=True, reason="overflow_error", metadata={"stage": stage_value}
            )

        if stage is CompactionStage.MID_TURN:
            apply_mid_turn = options.token_limit_reached and options.needs_follow_up
            return CompactionDecision(
                apply=apply_mid_turn,
                reason="mid_turn_follow_up" if apply_mid_turn else "mid_turn_not_required",
                metadata={
                    "stage": stage_value,
                    "token_limit_reached": options.token_limit_reached,
                    "needs_follow_up": options.needs_follow_up,
                },
            )

        # Pre-call auto-compaction
        total_tokens = llm_config.context_window_tokens
        used_tokens = self._estimate_tokens_from_messages(messages, options.chars_per_token)
        configured_limit = options.model_auto_compact_token_limit

        window_limit = None
        if total_tokens is not None:
            window_limit = int(max(total_tokens * options.auto_compact_ratio, 0))

        auto_compact_limit = configured_limit
        if auto_compact_limit is None:
            auto_compact_limit = window_limit
        elif window_limit is not None:
            auto_compact_limit = min(auto_compact_limit, window_limit)

        if auto_compact_limit is None or auto_compact_limit <= 0:
            return CompactionDecision(
                apply=False,
                reason="no_compaction_limit",
                metadata={
                    "stage": stage_value,
                    "total_tokens": total_tokens,
                    "used_tokens": used_tokens,
                },
            )

        apply_pre_turn = used_tokens >= auto_compact_limit
        return CompactionDecision(
            apply=apply_pre_turn,
            reason="auto_threshold" if apply_pre_turn else "below_threshold",
            metadata={
                "stage": stage_value,
                "total_tokens": total_tokens,
                "used_tokens": used_tokens,
                "auto_compact_limit": auto_compact_limit,
                "budget_used_tokens": budget.used_tokens,
            },
        )

    @override
    async def summarize(
        self,
        *,
        messages: Sequence[InternalMessage],
        llm_config: LLMConfig,
        llm_provider: LLMProvider,
        options: CompactorRunOptions,
    ) -> SummaryBuildResult:
        """调用模型生成"上下文交接摘要"。"""
        prompt = options.summarization_prompt or SUMMARIZATION_PROMPT
        base_messages = list(messages)
        retries = 0
        trimmed = 0

        while True:
            try:
                prompt_msg = InternalMessage(
                    info=UserMessageInfo(
                        id="compaction_req",
                        session_id="",
                        agent="",
                    ),
                    parts=[
                        TextPart(
                            session_id="",
                            message_id="compaction_req",
                            text=prompt,
                        )
                    ],
                )
                response = await llm_provider.complete(
                    messages=base_messages + [prompt_msg],
                    tools=None,
                )
                summary_text = self._extract_summary_text(response)
                if not summary_text:
                    return SummaryBuildResult.failure(
                        error="empty_summary",
                        trimmed_count=trimmed,
                        retries=retries,
                    )
                return SummaryBuildResult.success(
                    summary_text=summary_text,
                    trimmed_count=trimmed,
                    retries=retries,
                )
            except Exception as exc:
                if self._is_context_overflow_error(exc):
                    if len(base_messages) > 1 and trimmed < options.summary_max_trims:
                        _ = base_messages.pop(0)
                        trimmed += 1
                        continue
                    # 已无法继续裁剪，直接放弃重试
                    return SummaryBuildResult.failure(
                        error=f"context_overflow_untrimable: {type(exc).__name__}: {exc}",
                        trimmed_count=trimmed,
                        retries=retries,
                    )

                if retries >= options.summary_max_retries:
                    return SummaryBuildResult.failure(
                        error=f"{type(exc).__name__}: {exc}",
                        trimmed_count=trimmed,
                        retries=retries,
                    )

                retries += 1
                delay_s = (options.summary_retry_sleep_ms / 1000.0) * (2.0 ** (retries - 1))
                await asyncio.sleep(min(delay_s, 30.0))

    def _estimate_tokens_from_messages(
        self,
        messages: Sequence[InternalMessage],
        chars_per_token: float,
    ) -> int:
        """基于字符数的轻量 token 估算。"""
        total_chars = 0
        for message in messages:
            for part in message.parts:
                if isinstance(part, TextPart):
                    total_chars += len(part.text)
                    continue

                if isinstance(part, ToolPart):
                    arguments = part.state.input.arguments
                    if arguments:
                        total_chars += len(arguments)

                    if part.state.status == "completed":
                        total_chars += len(part.state.summary or part.state.output)
                        continue

                    if part.state.status == "error":
                        total_chars += len(part.state.error)
        return estimate_tokens(total_chars, chars_per_token)

    @staticmethod
    def _extract_summary_text(response: LLMStreamChunk | None) -> str:
        if response is None:
            return ""
        return response.content.strip() if response.content else ""

    @staticmethod
    def _is_context_overflow_error(exc: Exception) -> bool:
        return _is_context_overflow(exc)
