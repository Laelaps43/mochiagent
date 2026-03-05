"""
Agent Event Loop - Agent事件循环
结合事件驱动和Loop循环的优势：
- 外层：事件驱动，支持多人并发
- 内层：Loop循环，自动处理多轮对话
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from loguru import logger

from agent.core.bus import MessageBus
from agent.core.llm import AdapterRegistry
from agent.core.llm.errors import LLMProviderError
from agent.core.message import ToolPart
from agent.core.session import SessionManager
from agent.core.tools import ToolResultPostProcessor
from agent.types import ContextBudget, Event, EventType, SessionState

from .llm_turn import run_llm_turn

if TYPE_CHECKING:
    from agent.framework import AgentFramework


class AgentEventLoop:
    """
    Agent事件循环

    架构特点：
    1. 外层事件驱动 - 支持多session并发，自动排队
    2. 内层Loop循环 - 简化逻辑，自动处理tool_calls
    3. 并发控制 - 通过MessageBus的Semaphore控制最大并发数
    """

    def __init__(
        self,
        bus: MessageBus,
        session_manager: SessionManager,
        adapter_registry: AdapterRegistry,
        framework: AgentFramework,
        max_iterations: int = 100,
    ):
        self.bus = bus
        self.session_manager = session_manager
        self.adapter_registry = adapter_registry
        self.framework = framework
        self.max_iterations = max(1, max_iterations)
        self.tool_result_postprocessor = ToolResultPostProcessor()

        self.bus.subscribe(EventType.MESSAGE_RECEIVED, self._handle_user_message)
        logger.info(
            "AgentEventLoop initialized (max_iterations={})",
            self.max_iterations,
        )

    async def _emit_event(self, event: Event) -> None:
        if event.session_id:
            await self.session_manager.emit_to_session_listeners(event.session_id, event)

    async def _emit_error_and_done(
        self,
        *,
        session_id: str,
        error_message: str,
        code: str | None = None,
        hint: str | None = None,
    ) -> None:
        error_data: dict[str, str] = {"error": error_message}
        if code:
            error_data["code"] = code
        if hint:
            error_data["hint"] = hint

        await self._emit_event(
            Event(
                type=EventType.LLM_ERROR,
                session_id=session_id,
                data=error_data,
            )
        )

        message_id = None
        try:
            context = await self.session_manager.get_session(session_id)
            if context.current_message:
                message_id = context.current_message.message_id
        except Exception:
            pass

        await self._emit_event(
            Event(
                type=EventType.MESSAGE_DONE,
                session_id=session_id,
                data={
                    "message_id": message_id,
                    "cost": 0.0,
                    "tokens": {},
                    "context_budget": (await self._current_context_budget(session_id)).to_dict(),
                    "finish": "error",
                },
            )
        )

    @staticmethod
    def _extract_provider_error(exc: Exception) -> LLMProviderError | None:
        current: BaseException | None = exc
        while current:
            if isinstance(current, LLMProviderError):
                return current
            current = current.__cause__
        return None

    @classmethod
    def _resolve_error_payload(
        cls,
        exc: Exception,
    ) -> tuple[str, str | None, str | None]:
        provider_error = cls._extract_provider_error(exc)
        if provider_error:
            return provider_error.message, provider_error.code, provider_error.hint
        return f"{type(exc).__name__}: {exc}", None, None

    async def _persist_session_metadata(self, session_id: str) -> None:
        await self.session_manager.save_session_metadata(session_id)

    async def _current_context_budget(self, session_id: str) -> ContextBudget:
        try:
            context = await self.session_manager.get_session(session_id)
        except Exception:
            return ContextBudget()

        budget = context.context_budget
        if budget is None:
            return ContextBudget()
        return budget

    async def _handle_user_message(self, event: Event) -> None:
        session_id = event.session_id
        logger.info("Handling user message for session {}", session_id)

        try:
            context = await self.session_manager.get_session(session_id)
            if not context.messages or context.messages[-1].role != "user":
                raise ValueError(
                    f"No user message found in context for session {session_id}. "
                    "Please call context.build_user_message() first."
                )

            await self._conversation_loop(session_id)

        except ValueError as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            logger.error("Error handling user message (invalid state): {}", error_message)
            await self._emit_error_and_done(
                session_id=session_id,
                error_message=error_message,
                code="INVALID_STATE",
            )

        except Exception as exc:
            error_message, code, hint = self._resolve_error_payload(exc)
            logger.exception(
                "Error handling user message for session {}: {}",
                session_id,
                error_message,
            )
            await self._emit_error_and_done(
                session_id=session_id,
                error_message=error_message,
                code=code,
                hint=hint,
            )

    async def _conversation_loop(self, session_id: str) -> None:
        context = await self.session_manager.get_session(session_id)
        await self.session_manager.update_state(session_id, SessionState.PROCESSING)

        logger.info("Starting conversation loop for session {}", session_id)

        try:
            iteration_count = 0
            while iteration_count < self.max_iterations:
                iteration_count += 1
                result = await self._call_llm_turn(session_id)

                if result.get("tool_calls") and result.get("finish_reason") == "tool_calls":
                    tool_calls = result["tool_calls"]
                    logger.info("LLM requested {} tool calls", len(tool_calls))

                    await self.session_manager.update_state(session_id, SessionState.WAITING_TOOL)
                    await self._execute_tools(session_id, tool_calls)
                    await self.session_manager.finish_assistant_message(
                        session_id=session_id,
                        cost=result.get("cost", 0.0),
                        tokens=result.get("tokens", {}),
                        finish=result.get("finish_reason", "tool_calls"),
                    )
                    await self.session_manager.update_state(session_id, SessionState.PROCESSING)
                    continue

                logger.info(
                    "Conversation completed for session {} (finish_reason={})",
                    session_id,
                    result.get("finish_reason"),
                )

                result_budget = result.get("context_budget")
                if isinstance(result_budget, ContextBudget):
                    context_budget_payload = result_budget.to_dict()
                else:
                    context_budget_payload = (await self._current_context_budget(session_id)).to_dict()

                await self._emit_event(
                    Event(
                        type=EventType.MESSAGE_DONE,
                        session_id=session_id,
                        data={
                            "message_id": result.get("message_id"),
                            "cost": result.get("cost", 0.0),
                            "tokens": result.get("tokens", {}),
                            "context_budget": context_budget_payload,
                            "finish": result.get("finish_reason", "stop"),
                        },
                    )
                )
                return

            logger.error(
                "Max iterations exceeded for session {}: limit={}",
                session_id,
                self.max_iterations,
            )
            await self._emit_event(
                Event(
                    type=EventType.LLM_ERROR,
                    session_id=session_id,
                    data={
                        "error": (
                            "Conversation stopped: maximum iterations exceeded "
                            f"({self.max_iterations})"
                        ),
                        "code": "MAX_ITERATIONS_EXCEEDED",
                        "max_iterations": self.max_iterations,
                        "iterations": iteration_count,
                    },
                )
            )
            await self._emit_event(
                Event(
                    type=EventType.MESSAGE_DONE,
                    session_id=session_id,
                    data={
                        "message_id": (context.current_message.message_id if context.current_message else None),
                        "cost": 0.0,
                        "tokens": {},
                        "context_budget": (await self._current_context_budget(session_id)).to_dict(),
                        "finish": "max_iterations_exceeded",
                    },
                )
            )

        except Exception:
            await self.session_manager.update_state(session_id, SessionState.ERROR)
            raise

        finally:
            await self.session_manager.update_state(session_id, SessionState.IDLE)

    async def _call_llm_turn(self, session_id: str) -> dict:
        return await run_llm_turn(self, session_id)

    async def _execute_tools(self, session_id: str, tool_calls: list) -> list:
        context = await self.session_manager.get_session(session_id)
        message_id = context.current_message.message_id if context.current_message else None

        if not message_id:
            logger.warning("No current message for tool calls")
            return []

        agent = self.framework.get_agent(context.agent_name)
        if not agent:
            logger.error("Agent '{}' not found", context.agent_name)
            return []

        tool_parts_map: dict[str, ToolPart] = {}
        for tool_call in tool_calls:
            tool_part = ToolPart.create_running(session_id, message_id, tool_call)
            tool_parts_map[tool_call["id"]] = tool_part
            context.add_part_to_current(tool_part)

            await self._emit_event(
                Event(
                    type=EventType.PART_CREATED,
                    session_id=session_id,
                    data=tool_part.to_event_payload(),
                )
            )

        results = await agent.tool_executor.execute_batch(tool_calls)
        tool_args_by_call_id: dict[str, dict] = {}
        for tool_call in tool_calls:
            function_info = tool_call.get("function", {})
            call_id = tool_call.get("id", "")
            raw_args = function_info.get("arguments", "{}")
            try:
                parsed_args = json.loads(raw_args)
            except Exception:
                parsed_args = {"raw": raw_args}
            tool_args_by_call_id[call_id] = parsed_args

        for result in results:
            call_id = result.tool_call_id
            if call_id not in tool_parts_map:
                continue

            original_part = tool_parts_map[call_id]
            if result.success:
                processed_result = await self.tool_result_postprocessor.process(
                    session_id=session_id,
                    tool_result=result,
                    tool_arguments=tool_args_by_call_id.get(call_id, {}),
                    storage=self.session_manager.storage,
                )
                updated_part = original_part.update_to_completed(processed_result)
            else:
                updated_part = original_part.update_to_error(result)

            for i, part in enumerate(context.current_message.parts):
                if isinstance(part, ToolPart) and part.call_id == call_id:
                    context.current_message.parts[i] = updated_part
                    break

            await self._emit_event(
                Event(
                    type=EventType.PART_CREATED,
                    session_id=session_id,
                    data=updated_part.to_event_payload(),
                )
            )

        return results
