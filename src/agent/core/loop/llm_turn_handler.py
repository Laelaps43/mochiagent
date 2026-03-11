"""LLM turn execution helper for AgentEventLoop."""

from __future__ import annotations

from copy import deepcopy
import time
from typing import Awaitable, Callable, TYPE_CHECKING

from loguru import logger

from agent.core.compression import CompactionPayload, CompactionStage
from agent.core.llm import AdapterRegistry
from agent.core.llm.errors import LLMProviderError, is_context_overflow_error as _is_context_overflow
from agent.core.loop.turn_result import LLMTurnResult
from agent.core.message import Message as InternalMessage, ReasoningPart, TextPart
from agent.core.message.info import AssistantMessageInfo
from agent.core.runtime import StrategyKind
from agent.core.session import SessionManager
from agent.core.utils import extract_turn_tokens
from agent.types import ContextBudget, Event, EventType, LLMConfig, ProviderUsage, TokenUsage, ToolCallPayload

if TYPE_CHECKING:
    from agent.core.llm.base import LLMProvider
    from agent.core.runtime.strategy_manager import AgentStrategyManager
    from agent.core.session.context import SessionContext
    from agent.framework import AgentFramework


class LLMTurnHandler:
    def __init__(
        self,
        *,
        session_manager: SessionManager,
        adapter_registry: AdapterRegistry,
        framework: AgentFramework,
        emit_event: Callable[[Event], Awaitable[None]],
    ) -> None:
        self.session_manager = session_manager
        self.adapter_registry = adapter_registry
        self.framework = framework
        self._emit_event = emit_event

    async def run(self, session_id: str) -> LLMTurnResult:
        context = await self.session_manager.get_session(session_id)

        if not context.model_profile_id:
            raise ValueError(
                f"Session {session_id} has no model_profile_id. "
                "Please take_session with a valid model_profile_id first."
            )

        agent = self.framework.get_agent(context.agent_name)
        if agent is None:
            raise ValueError(f"Agent '{context.agent_name}' not found")

        llm_config = agent.context.resolve_llm_config_for_agent(
            context.agent_name,
            context.model_profile_id,
        )
        llm = self.adapter_registry.get(llm_config)
        compaction_events: list[CompactionPayload] = []

        pre_compaction = await self._run_context_compaction(
            context=context,
            budget=deepcopy(context.context_budget),
            llm_config=llm_config,
            llm=llm,
            strategy_manager=agent.context.strategy_manager,
            stage=CompactionStage.PRE_CALL,
        )
        compaction_events.append(pre_compaction)
        if pre_compaction.applied:
            await self._persist_session_metadata(session_id)

        # 复用已有的 assistant message（tool_calls 轮次），或创建新的
        if context.current_message and isinstance(context.current_message.info, AssistantMessageInfo):
            assistant_msg = context.current_message
        else:
            last_user_msg = context.messages[-1]
            assistant_msg = context.build_assistant_message(
                parent_id=last_user_msg.message_id,
                provider_id=llm_config.provider,
                model_id=llm_config.model,
            )
        message_id = assistant_msg.message_id

        provider_usage: ProviderUsage | None = None

        tools = agent.tool_registry.get_definitions()
        system_prompt = agent.get_system_prompt(context)

        overflow_retries = 0
        max_overflow_retries = llm_config.max_overflow_retries
        text_buffer = ""
        thinking_buffer = ""
        accumulated_tool_calls: list[ToolCallPayload] = []
        finish_reason = None

        while True:
            reasoning_buffer = ""
            reasoning_start_time = None
            text_buffer = ""
            thinking_buffer = ""
            accumulated_tool_calls = []
            finish_reason = None
            provider_usage = None

            llm_messages = (
                [InternalMessage.create_system(system_prompt)] + list(context.messages)
                if system_prompt
                else list(context.messages)
            )
            logger.debug(f"Calling LLM for session {session_id}: messages={len(llm_messages)}")

            try:
                async for chunk in llm.stream_chat(
                    messages=llm_messages,
                    tools=tools,
                ):
                    if chunk.thinking:
                        reasoning_buffer += chunk.thinking
                        thinking_buffer += chunk.thinking
                        if reasoning_start_time is None:
                            reasoning_start_time = int(time.time() * 1000)
                        await self._emit_event(
                            Event(
                                type=EventType.LLM_THINKING,
                                session_id=session_id,
                                data={
                                    "message_id": message_id,
                                    "thinking": chunk.thinking,
                                },
                            )
                        )

                    if chunk.content:
                        if reasoning_buffer:
                            reasoning_part = ReasoningPart.create_fast(
                                session_id=session_id,
                                message_id=message_id,
                                text=reasoning_buffer,
                                start=reasoning_start_time or int(time.time() * 1000),
                                end=int(time.time() * 1000),
                            )
                            context.add_part_to_current(reasoning_part)
                            await self._emit_event(
                                Event(
                                    type=EventType.PART_CREATED,
                                    session_id=session_id,
                                    data=reasoning_part.to_event_payload(),
                                )
                            )
                            reasoning_buffer = ""

                        text_buffer += chunk.content
                        await self._emit_event(
                            Event(
                                type=EventType.PART_CREATED,
                                session_id=session_id,
                                data={
                                    "type": "text",
                                    "text": chunk.content,
                                    "session_id": session_id,
                                    "message_id": message_id,
                                },
                            )
                        )

                    if chunk.tool_calls:
                        accumulated_tool_calls.extend(chunk.tool_calls)
                    if chunk.finish_reason:
                        finish_reason = chunk.finish_reason
                    if chunk.usage is not None:
                        provider_usage = chunk.usage

                if reasoning_buffer:
                    reasoning_part = ReasoningPart.create_fast(
                        session_id=session_id,
                        message_id=message_id,
                        text=reasoning_buffer,
                        start=reasoning_start_time or int(time.time() * 1000),
                        end=int(time.time() * 1000),
                    )
                    context.add_part_to_current(reasoning_part)
                    await self._emit_event(
                        Event(
                            type=EventType.PART_CREATED,
                            session_id=session_id,
                            data=reasoning_part.to_event_payload(),
                        )
                    )

                # 流结束后，将累积的文本作为单个 TextPart 添加到消息
                if text_buffer:
                    final_text_part = TextPart.create_fast(
                        session_id=session_id,
                        message_id=message_id,
                        text=text_buffer,
                    )
                    context.add_part_to_current(final_text_part)

                break
            except Exception as exc:
                if overflow_retries >= max_overflow_retries or not self.is_context_overflow_error(exc):
                    raise

                overflow_retries += 1
                overflow_compaction = await self._run_context_compaction(
                    context=context,
                    budget=deepcopy(context.context_budget),
                    llm_config=llm_config,
                    llm=llm,
                    strategy_manager=agent.context.strategy_manager,
                    stage=CompactionStage.OVERFLOW_ERROR,
                    error=str(exc),
                )
                compaction_events.append(overflow_compaction)
                if not overflow_compaction.applied:
                    raise
                if context.current_message:
                    context.current_message.parts = []
                await self._persist_session_metadata(session_id)
                continue

        turn_tokens, source = extract_turn_tokens(provider_usage)
        assistant_msg.info.tokens.input += turn_tokens.input
        assistant_msg.info.tokens.output += turn_tokens.output
        assistant_msg.info.tokens.reasoning += turn_tokens.reasoning

        context_budget: ContextBudget = context.update_context_budget(
            total_tokens=llm_config.context_window_tokens,
            input_tokens=turn_tokens.input,
            output_tokens=turn_tokens.output,
            reasoning_tokens=turn_tokens.reasoning,
            source=source,
        )
        await self._persist_session_metadata(session_id)


        last_compaction = (
            compaction_events[-1]
            if compaction_events
            else CompactionPayload.invalid(stage=CompactionStage.PRE_CALL.value)
        )
        return LLMTurnResult(
            content=text_buffer,
            thinking=thinking_buffer,
            tool_calls=accumulated_tool_calls,
            finish_reason=finish_reason,
            tokens=assistant_msg.info.tokens,
            context_budget=context_budget,
            context_compaction=last_compaction,
            context_compaction_events=compaction_events,
            message_id=message_id,
        )

    @staticmethod
    def extract_provider_error(exc: Exception) -> LLMProviderError | None:
        current: BaseException | None = exc
        while current:
            if isinstance(current, LLMProviderError):
                return current
            current = current.__cause__
        return None

    @classmethod
    def resolve_error_payload(
        cls,
        exc: Exception,
    ) -> tuple[str, str | None, str | None]:
        provider_error = cls.extract_provider_error(exc)
        if provider_error:
            return provider_error.message, provider_error.code, provider_error.hint
        return f"{type(exc).__name__}: {exc}", None, None

    @staticmethod
    def is_context_overflow_error(exc: Exception) -> bool:
        return _is_context_overflow(exc)

    async def _persist_session_metadata(self, session_id: str) -> None:
        await self.session_manager.save_session_metadata(session_id)

    async def _run_context_compaction(
        self,
        *,
        context: SessionContext,
        budget: ContextBudget,
        llm_config: LLMConfig,
        llm: LLMProvider,
        strategy_manager: AgentStrategyManager,
        stage: CompactionStage,
        error: str | None = None,
    ) -> CompactionPayload:
        result = await strategy_manager.run(
            StrategyKind.CONTEXT_COMPACTION,
            session_context=context,
            budget=budget,
            llm_config=llm_config,
            llm_provider=llm,
            agent_name=context.agent_name,
            stage=stage,
            error=error,
        )
        if isinstance(result, CompactionPayload):
            return result
        return CompactionPayload.invalid(stage=stage.value)
