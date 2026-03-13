"""Context compaction interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, override

from loguru import logger

from .stage import CompactionStage
from .types import (
    CompactorRunOptions,
    CompactionDecision,
    CompactionPayload,
    SummaryBuildResult,
)

if TYPE_CHECKING:
    from agent.core.llm import LLMProvider
    from agent.core.message import Message
    from agent.core.session.context import SessionContext
    from agent.types import ContextBudget, Event, LLMConfig

EmitEvent = Callable[["Event"], Awaitable[None]]


class ContextCompactor(ABC):
    """Extension point: decide whether to compact and how to generate a summary.

    Implementations only override ``should_compact`` and ``summarize`` — they
    receive a read-only message sequence and never touch ``SessionContext``.

    The concrete ``run`` method (template method) orchestrates the full flow:
    get visible messages → decide → summarize → insert bookmark.
    """

    # ------------------------------------------------------------------
    # Template method — concrete, not meant to be overridden
    # ------------------------------------------------------------------

    async def run(
        self,
        *,
        session_context: "SessionContext",
        budget: "ContextBudget",
        llm_config: "LLMConfig",
        llm_provider: "LLMProvider",
        stage: CompactionStage,
        options: CompactorRunOptions,
        emit_event: EmitEvent | None = None,
    ) -> CompactionPayload:
        """Orchestrate the compaction flow.

        1. Get LLM-visible messages from the session context.
        2. Ask ``should_compact`` whether compaction is needed.
        3. Emit CONTEXT_COMPACTING event.
        4. Call ``summarize`` to produce a summary.
        5. Compute bookmark position and insert a ``CompactionMessage``.
        6. Emit CONTEXT_COMPACTED event.
        """
        from agent.core.message import Message as InternalMessage
        from agent.types import Event, EventType

        name = type(self).__name__
        stage_value = stage.value
        session_id = session_context.session_id

        try:
            llm_messages = session_context.get_llm_messages()

            decision = self.should_compact(
                messages=llm_messages,
                budget=budget,
                llm_config=llm_config,
                stage=stage,
                options=options,
            )
            if not decision.apply:
                return CompactionPayload(
                    applied=False, reason=decision.reason, name=name, stage=stage_value
                )

            # 开始压缩
            if emit_event:
                await emit_event(
                    Event(
                        type=EventType.CONTEXT_COMPACTING,
                        session_id=session_id,
                        data={"reason": decision.reason, "stage": stage_value},
                    )
                )

            summary_result = await self.summarize(
                messages=llm_messages,
                llm_config=llm_config,
                llm_provider=llm_provider,
                options=options,
            )
            if not summary_result.ok:
                return CompactionPayload(
                    applied=False, reason="summary_failed", name=name, stage=stage_value
                )

            insert_idx = self.compute_bookmark_position(
                session_context.messages,
                options.keep_user_tokens_budget,
                options.chars_per_token,
            )

            if insert_idx == 0:
                logger.warning(
                    "Compaction bookmark position is 0 — no messages to compact, skipping"
                )
                return CompactionPayload(
                    applied=False,
                    reason="no_messages_to_compact",
                    name=name,
                    stage=stage_value,
                )

            bookmark = InternalMessage.create_compaction(
                session_id=session_id,
                summary=summary_result.summary_text,
                compacted_count=insert_idx,
                compaction_metadata=decision.metadata,
            )
            session_context.apply_compaction(bookmark, insert_idx)

            # 压缩完成
            if emit_event:
                await emit_event(
                    Event(
                        type=EventType.CONTEXT_COMPACTED,
                        session_id=session_id,
                        data={
                            "last_compaction_message_id": bookmark.message_id,
                            "reason": decision.reason,
                            "stage": stage_value,
                        },
                    )
                )

            return CompactionPayload(
                applied=True, reason=decision.reason, name=name, stage=stage_value
            )
        except Exception as exc:
            logger.exception(
                "Context compactor failed at stage '{}': {}",
                stage_value,
                exc,
            )
            return CompactionPayload(
                applied=False,
                reason=f"compactor_error: {type(exc).__name__}",
                name=name,
                stage=stage_value,
            )

    # ------------------------------------------------------------------
    # Abstract extension points — subclasses implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def should_compact(
        self,
        *,
        messages: Sequence["Message"],
        budget: "ContextBudget",
        llm_config: "LLMConfig",
        stage: CompactionStage,
        options: CompactorRunOptions,
    ) -> CompactionDecision:
        """Decide whether compaction should be applied."""
        raise NotImplementedError

    @abstractmethod
    async def summarize(
        self,
        *,
        messages: Sequence["Message"],
        llm_config: "LLMConfig",
        llm_provider: "LLMProvider",
        options: CompactorRunOptions,
    ) -> SummaryBuildResult:
        """Generate a context summary from the given messages."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_bookmark_position(
        messages: list["Message"],
        keep_user_budget: int,
        chars_per_token: float,
    ) -> int:
        """从消息末尾往前遍历，按 token 预算计算保留多少条最近消息，返回书签插入位置。"""
        from agent.core.message import TextPart, ReasoningPart
        from agent.core.utils import estimate_tokens

        used = 0
        boundary = len(messages)

        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            msg_chars = 0
            for part in msg.parts:
                if isinstance(part, (TextPart, ReasoningPart)):
                    msg_chars += len(part.text)
            msg_tokens = estimate_tokens(msg_chars, chars_per_token)

            if used + msg_tokens > keep_user_budget:
                break
            used += msg_tokens
            boundary = idx

        return boundary


class NoopContextCompactor(ContextCompactor):
    """Default compactor that never compacts."""

    @override
    def should_compact(
        self,
        *,
        messages: Sequence["Message"],
        budget: "ContextBudget",
        llm_config: "LLMConfig",
        stage: CompactionStage,
        options: CompactorRunOptions,
    ) -> CompactionDecision:
        return CompactionDecision(apply=False, reason="noop")

    @override
    async def summarize(
        self,
        *,
        messages: Sequence["Message"],
        llm_config: "LLMConfig",
        llm_provider: "LLMProvider",
        options: CompactorRunOptions,
    ) -> SummaryBuildResult:
        return SummaryBuildResult.failure(error="noop")
