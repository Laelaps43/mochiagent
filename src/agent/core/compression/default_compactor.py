"""Default context compactor implementation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, override

from agent.core.llm.errors import is_context_overflow_error as _is_context_overflow
from agent.types import ContextBudget, LLMConfig, LLMStreamChunk

from agent.core.compression.compactor import ContextCompactor
from agent.core.compression.types import (
    CompactorRunOptions,
    CompactionDecision,
    CompactionResult,
    RewriteStats,
    SummaryBuildResult,
)
from agent.core.message import (
    Message as InternalMessage,
    ReasoningPart,
    TextPart,
    ToolPart,
    UserMessageInfo,
    UserTextInput,
)
from agent.core.compression.stage import CompactionStage
from agent.core.utils import estimate_tokens

if TYPE_CHECKING:
    from agent.core.llm.base import LLMProvider
    from agent.core.session.context import SessionContext


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
    async def run(
        self,
        *,
        session_context: SessionContext,
        budget: ContextBudget,
        llm_config: LLMConfig,
        llm_provider: LLMProvider,
        stage: CompactionStage,
        error: str | None = None,
        options: CompactorRunOptions,
    ) -> CompactionResult:
        """执行默认压缩策略。

        入口约定：
        - `stage` 由外层事件循环传入，表示当前调用时机（pre_call/overflow_error/mid_turn）。
        - `budget` 为当前已知预算快照；若 provider 不返回 usage，则这里通常是估算值。
        - 方法可以直接修改 `session_context.messages`，调用方会在外层负责持久化。

        返回语义：
        - `applied=False`：本次未触发压缩或压缩失败（不会抛异常中断主流程）。
        - `applied=True`：上下文已被重写为"最近用户消息 + 摘要"结构。
        """
        stage_value = stage.value

        # 先判断是否触发压缩；不触发直接返回，避免做额外 LLM 调用
        trigger = self._should_compact(
            session_context=session_context,
            budget=budget,
            llm_config=llm_config,
            stage=stage,
            options=options,
        )
        if not trigger.apply:
            return CompactionResult(
                applied=False,
                reason=trigger.reason,
                metadata=trigger.metadata,
            )

        # 触发后先生成摘要（默认 inline：调用同一个 provider 的 complete）
        prompt = options.summarization_prompt or SUMMARIZATION_PROMPT
        summary_result = await self._build_summary(
            session_context=session_context,
            llm_provider=llm_provider,
            prompt=prompt,
            max_retries=options.summary_max_retries,
            max_trims=options.summary_max_trims,
            retry_sleep_ms=options.summary_retry_sleep_ms,
        )
        if not summary_result.ok:
            return CompactionResult(
                applied=False,
                reason="summary_generation_failed",
                metadata={
                    "stage": stage_value,
                    "error": summary_result.error,
                    "trigger": trigger.metadata,
                },
            )

        summary_text = summary_result.summary_text

        # 用"保留最近用户消息 + 摘要"重写上下文
        rewritten = self._rewrite_messages(
            session_context=session_context,
            summary_text=summary_text,
            keep_user_budget=options.keep_user_tokens_budget,
            chars_per_token=options.chars_per_token,
        )
        if rewritten.before_messages == rewritten.after_messages:
            return CompactionResult(
                applied=False,
                reason="no_effect",
                metadata={
                    "stage": stage_value,
                    "trigger": trigger.metadata,
                    "summary_trimmed_count": summary_result.trimmed_count,
                },
            )

        return CompactionResult(
            applied=True,
            reason=trigger.reason,
            metadata={
                "stage": stage_value,
                "trigger": trigger.metadata,
                "summary_trimmed_count": summary_result.trimmed_count,
                "summary_tokens_estimated": estimate_tokens(summary_text, options.chars_per_token),
                "summary_request_retries": summary_result.retries,
            },
            stats={
                "before_messages": rewritten.before_messages,
                "after_messages": rewritten.after_messages,
                "retained_user_messages": rewritten.retained_user_messages,
                "dropped_messages": rewritten.dropped_messages,
                "truncated_messages": rewritten.truncated_messages,
            },
        )

    def _should_compact(
        self,
        *,
        session_context: SessionContext,
        budget: ContextBudget,
        llm_config: LLMConfig,
        stage: CompactionStage,
        options: CompactorRunOptions,
    ) -> CompactionDecision:
        """决定"是否压缩"以及"触发原因"。

        设计目标：
        - 触发判定与压缩执行解耦，方便后续替换判定逻辑（例如更复杂的预算器）。
        - 所有分支都返回结构化 metadata，便于事件上报与调试。
        """
        stage_value = stage.value

        # Provider returned a context-overflow error, so compaction is mandatory.
        if stage is CompactionStage.OVERFLOW_ERROR:
            return CompactionDecision(
                apply=True, reason="overflow_error", metadata={"stage": stage_value}
            )

        if stage is CompactionStage.MID_TURN:
            # Mid-turn compaction is reserved for "continue generation" style flows:
            # 1) token limit reached in current completion
            # 2) caller decides another follow-up turn is needed
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

        # Pre-call is the default auto-compaction checkpoint.
        total_tokens = llm_config.context_window_tokens
        # 注意：这里使用"会话文本长度估算"而不是 budget.used_tokens。
        # 原因是 budget 可能来自上轮或 provider usage，不能完全代表"当前可发送历史"的体积。
        used_tokens = self._estimate_tokens_from_context(session_context, options.chars_per_token)
        configured_limit = options.model_auto_compact_token_limit

        # Effective threshold:
        # min(model_auto_compact_token_limit, context_window * auto_compact_ratio)
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

    async def _build_summary(
        self,
        *,
        session_context: SessionContext,
        llm_provider: LLMProvider,
        prompt: str,
        max_retries: int,
        max_trims: int,
        retry_sleep_ms: int,
    ) -> SummaryBuildResult:
        """调用模型生成"上下文交接摘要"。

        返回字段：
        - ok: 是否成功获得摘要文本
        - summary_text: 摘要内容（仅 ok=True 时存在）
        - trimmed_count: 因"压缩请求本身超窗"而裁掉的最老消息条数
        - retries: 非超窗错误的重试次数
        """
        base_messages = list(session_context.messages)
        retries = 0
        trimmed = 0

        while True:
            try:
                prompt_msg = InternalMessage(
                    info=UserMessageInfo(
                        id="compaction_req",
                        session_id=session_context.session_id,
                        agent=session_context.agent_name,
                    ),
                    parts=[
                        TextPart.create_fast(
                            session_id=session_context.session_id,
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
                if (
                    self._is_context_overflow_error(exc)
                    and len(base_messages) > 1
                    and trimmed < max_trims
                ):
                    _ = base_messages.pop(0)
                    trimmed += 1
                    continue

                if retries >= max_retries:
                    return SummaryBuildResult.failure(
                        error=f"{type(exc).__name__}: {exc}",
                        trimmed_count=trimmed,
                        retries=retries,
                    )

                retries += 1
                await asyncio.sleep((retry_sleep_ms / 1000.0) * (2.0 ** (retries - 1)))

    def _rewrite_messages(
        self,
        *,
        session_context: SessionContext,
        summary_text: str,
        keep_user_budget: int,
        chars_per_token: float,
    ) -> RewriteStats:
        """将历史重写为"可继续对话"的紧凑结构。

        重写规则：
        1. 按预算保留最近 user 消息（从新到旧挑选，直到预算耗尽）
        2. 追加一条 `COMPACTION_SUMMARY` 用户消息
        3. 保证"最后一条真实 user 消息"仍位于末尾，避免模型把摘要当成最新用户输入
        """
        messages = list(session_context.messages)
        before_messages = len(messages)
        if before_messages == 0:
            return RewriteStats(
                before_messages=0,
                after_messages=0,
                retained_user_messages=0,
                truncated_messages=0,
            )

        summary_prefix = SUMMARY_PREFIX
        last_user_index = self._find_last_user_index(messages)
        last_user_text = ""
        if last_user_index is not None:
            last_user_text = self._message_text(messages[last_user_index])

        # 从后往前保留 user 消息：优先最新；超过预算时截断最后一条并停止
        retained_texts: list[str] = []
        truncated_messages = 0
        used = 0
        for idx in range(len(messages) - 1, -1, -1):
            if idx == last_user_index:
                continue
            msg = messages[idx]
            if msg.role != "user":
                continue
            text = self._message_text(msg).strip()
            if not text:
                continue
            if text.startswith(summary_prefix):
                continue

            message_tokens = estimate_tokens(text, chars_per_token)
            if used + message_tokens <= keep_user_budget:
                retained_texts.append(text)
                used += message_tokens
                continue

            remaining = max(keep_user_budget - used, 0)
            if remaining > 0:
                truncated_messages += 1
                max_chars = int(remaining * chars_per_token)
                # 截断时保留尾部，倾向保留"最近上下文"而非历史开头
                tail_text = text[-max_chars:] if max_chars > 0 else ""
                if tail_text:
                    retained_texts.append(f"{tail_text}\n[tokens truncated]")
            break

        retained_texts.reverse()
        if last_user_text.strip():
            retained_texts.append(last_user_text.strip())

        # 重建消息：保留 user 片段 + summary（都以 user message 存储）
        session_context.messages = []
        session_context.current_message = None
        for text in retained_texts:
            _ = session_context.build_user_message(parts=[UserTextInput(text=text)])
        _ = session_context.build_user_message(
            parts=[UserTextInput(text=f"{summary_prefix}{summary_text.strip()}")]
        )

        # 关键约束：最新真实 user 需保持在末尾，避免下一轮对 summary 直接回复
        if last_user_text.strip():
            summary_message = session_context.messages.pop()
            session_context.messages.insert(
                max(len(session_context.messages) - 1, 0), summary_message
            )

        return RewriteStats(
            before_messages=before_messages,
            after_messages=len(session_context.messages),
            retained_user_messages=len(retained_texts),
            truncated_messages=truncated_messages,
        )

    @staticmethod
    def _message_text(message: InternalMessage) -> str:
        chunks: list[str] = []
        for part in message.parts:
            if isinstance(part, (TextPart, ReasoningPart)) and part.text:
                chunks.append(part.text)
        return "".join(chunks)

    @staticmethod
    def _find_last_user_index(messages: list[InternalMessage]) -> int | None:
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].info.role == "user":
                return idx
        return None

    def _estimate_tokens_from_context(
        self,
        session_context: SessionContext,
        chars_per_token: float,
    ) -> int:
        """基于字符数的轻量 token 估算。

        这是一个保守近似，不追求精准计费，只用于"是否需要压缩"的触发判断。
        """
        total_chars = 0
        # 性能优化：直接遍历 message/part，避免构建 llm_messages 中间结构。
        for message in session_context.messages:
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
