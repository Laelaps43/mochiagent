"""LLM turn execution helper for AgentEventLoop."""

from __future__ import annotations

from copy import deepcopy
import time
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from loguru import logger

from agent.core.compression import CompactionPayload, CompactionStage
from agent.core.llm import AdapterRegistry
from agent.core.llm.errors import LLMProviderError, is_context_overflow_error as _is_context_overflow
from agent.core.loop.turn_result import LLMTurnResult
from agent.core.message import ReasoningPart, TextPart
from agent.core.prompt import inject_system_prompt
from agent.core.runtime import StrategyKind
from agent.core.session import SessionManager
from agent.types import ContextBudget, Event, EventType, TokenUsage

if TYPE_CHECKING:
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

    @staticmethod
    def _empty_context_budget() -> ContextBudget:
        return ContextBudget()

    async def run(self, session_id: str) -> LLMTurnResult:
        context = await self.session_manager.get_session(session_id)

        if not context.model_profile_id:
            raise ValueError(
                f"Session {session_id} has no model_profile_id. "
                "Please take_session with a valid model_profile_id first."
            )

        llm_config = self.framework.resolve_llm_config_for_agent(
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
            stage=CompactionStage.PRE_CALL,
        )
        compaction_events.append(pre_compaction)
        if pre_compaction.applied:
            await self._persist_session_metadata(session_id)

        last_user_msg = context.messages[-1]
        assistant_msg = context.build_assistant_message(
            parent_id=last_user_msg.message_id,
            provider_id=llm_config.provider,
            model_id=llm_config.model,
        )
        message_id = assistant_msg.message_id

        total_tokens: TokenUsage = {"input": 0, "output": 0, "reasoning": 0}
        total_cost = 0.0
        provider_usage: dict | None = None

        agent = self.framework.get_agent(context.agent_name)
        tools = agent.tool_registry.get_definitions() if agent else []
        system_prompt = agent.get_system_prompt(context) if agent else None

        overflow_retries = 0
        max_overflow_retries = 1
        text_buffer = ""
        thinking_buffer = ""
        accumulated_tool_calls: list = []
        finish_reason = None

        while True:
            reasoning_buffer = ""
            reasoning_start_time = None
            text_buffer = ""
            thinking_buffer = ""
            accumulated_tool_calls = []
            finish_reason = None
            provider_usage = None

            llm_messages = inject_system_prompt(
                context.get_llm_messages(),
                system_prompt,
            )
            logger.debug(f"Calling LLM for session {session_id}: messages={len(llm_messages)}")

            try:
                async for chunk in llm.stream_chat(
                    messages=llm_messages,
                    tools=tools,
                ):
                    if "thinking" in chunk and chunk["thinking"]:
                        thinking_content = chunk["thinking"]
                        reasoning_buffer += thinking_content
                        thinking_buffer += thinking_content
                        if reasoning_start_time is None:
                            reasoning_start_time = int(time.time() * 1000)
                        await self._emit_event(
                            Event(
                                type=EventType.LLM_THINKING,
                                session_id=session_id,
                                data={
                                    "message_id": message_id,
                                    "thinking": thinking_content,
                                },
                            )
                        )

                    if "content" in chunk and chunk["content"]:
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

                        text_buffer += chunk["content"]
                        text_part = TextPart.create_fast(
                            session_id=session_id,
                            message_id=message_id,
                            text=chunk["content"],
                        )
                        context.add_part_to_current(text_part)
                        await self._emit_event(
                            Event(
                                type=EventType.PART_CREATED,
                                session_id=session_id,
                                data=text_part.to_event_payload(),
                            )
                        )

                    if "tool_calls" in chunk:
                        accumulated_tool_calls = chunk["tool_calls"]
                    if "finish_reason" in chunk:
                        finish_reason = chunk["finish_reason"]
                    if "usage" in chunk and isinstance(chunk["usage"], dict):
                        provider_usage = chunk["usage"]

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

        total_tokens, source = self.extract_turn_tokens_from_usage(provider_usage)
        context_budget: ContextBudget = context.update_context_budget(
            total_tokens=llm_config.context_window_tokens,
            input_tokens=total_tokens.get("input", 0),
            output_tokens=total_tokens.get("output", 0),
            reasoning_tokens=total_tokens.get("reasoning", 0),
            source=source,
        )
        await self._persist_session_metadata(session_id)

        if not accumulated_tool_calls:
            await self.session_manager.finish_assistant_message(
                session_id=session_id,
                cost=total_cost,
                tokens=total_tokens,
                finish=finish_reason or "stop",
            )

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
            cost=total_cost,
            tokens=total_tokens,
            context_budget=context_budget,
            context_compaction=last_compaction,
            context_compaction_events=compaction_events,
            message_id=message_id,
        )

    async def current_context_budget(self, session_id: str) -> ContextBudget:
        try:
            context = await self.session_manager.get_session(session_id)
        except Exception:
            return self._empty_context_budget()
        return context.context_budget

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

    @staticmethod
    def _token_to_int(value: object) -> int:
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return max(value, 0)
        if isinstance(value, float):
            return max(int(value), 0)
        return 0

    @classmethod
    def extract_turn_tokens_from_usage(cls, usage: dict | None) -> tuple[TokenUsage, str]:
        if not usage:
            return {"input": 0, "output": 0, "reasoning": 0}, "estimated"

        input_tokens = cls._token_to_int(
            usage.get("prompt_tokens", usage.get("input_tokens", 0))
        )
        output_tokens = cls._token_to_int(
            usage.get("completion_tokens", usage.get("output_tokens", 0))
        )
        reasoning_tokens = cls._token_to_int(usage.get("reasoning_tokens", 0))
        if reasoning_tokens == 0:
            details = usage.get("completion_tokens_details")
            if isinstance(details, dict):
                reasoning_tokens = cls._token_to_int(details.get("reasoning_tokens", 0))

        return {
            "input": input_tokens,
            "output": output_tokens,
            "reasoning": reasoning_tokens,
        }, "provider"

    async def _persist_session_metadata(self, session_id: str) -> None:
        await self.session_manager.save_session_metadata(session_id)

    async def _run_context_compaction(
        self,
        *,
        context: Any,
        budget: ContextBudget,
        llm_config: Any,
        llm: Any,
        stage: CompactionStage,
        error: str | None = None,
    ) -> CompactionPayload:
        result = await self.framework.strategy_manager.run(
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
