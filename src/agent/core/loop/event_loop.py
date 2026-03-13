"""
Agent Event Loop - Agent事件循环
结合事件驱动和Loop循环的优势：
- 外层：事件驱动，支持多人并发
- 内层：Loop循环，自动处理多轮对话
"""

from __future__ import annotations

import asyncio
import json
from typing import cast

from loguru import logger

from agent.core.bus import MessageBus
from agent.core.compression import CompactionPayload
from agent.core.llm import AdapterRegistry
from agent.core.loop.llm_turn_handler import LLMTurnHandler
from agent.core.message import ToolPart
from agent.core.session import SessionManager
from agent.types import (
    ContextBudget,
    Event,
    EventType,
    SessionState,
    TokenUsage,
    ToolCallPayload,
    ToolResult,
)

from agent.core.loop._framework_protocol import FrameworkProtocol


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
        framework: FrameworkProtocol,
        max_iterations: int = 100,
    ):
        self.bus: MessageBus = bus
        self.session_manager: SessionManager = session_manager
        self.adapter_registry: AdapterRegistry = adapter_registry
        self.framework: FrameworkProtocol = framework
        self.max_iterations: int = max(1, max_iterations)
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._session_locks_guard: asyncio.Lock = asyncio.Lock()
        self._active_tasks: dict[str, asyncio.Task[None]] = {}
        self._llm_turn_handler: LLMTurnHandler = LLMTurnHandler(
            session_manager=session_manager,
            adapter_registry=adapter_registry,
            framework=framework,
            emit_event=self._emit_event,
        )

        self.bus.subscribe(EventType.MESSAGE_RECEIVED, self._handle_user_message)
        self.bus.subscribe(EventType.SESSION_TERMINATED, self._handle_session_terminated)
        logger.info(
            "AgentEventLoop initialized (max_iterations={})",
            self.max_iterations,
        )

    async def _emit_event(self, event: Event) -> None:
        if event.session_id:
            await self.session_manager.emit_to_session_listeners(event.session_id, event)

    async def _handle_session_terminated(self, event: Event) -> None:
        session_id = event.session_id
        task = self._active_tasks.pop(session_id, None)
        if task is not None and not task.done():
            _ = task.cancel()
            logger.info("Cancelled active conversation task for session {}", session_id)
        await self.remove_session_lock(session_id)

    async def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        async with self._session_locks_guard:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._session_locks[session_id] = lock
            return lock

    async def remove_session_lock(self, session_id: str) -> None:
        async with self._session_locks_guard:
            _ = self._session_locks.pop(session_id, None)

    async def _try_cleanup_session_lock(self, session_id: str) -> None:
        """Remove the session lock if it is not currently held."""
        async with self._session_locks_guard:
            lock = self._session_locks.get(session_id)
            if lock is not None and not lock.locked():
                del self._session_locks[session_id]

    async def cleanup_stale_locks(self) -> None:
        """Remove locks for sessions that are no longer in the session manager cache."""
        active_ids = set(await self.session_manager.cached_session_ids())
        async with self._session_locks_guard:
            stale = [sid for sid in self._session_locks if sid not in active_ids]
            for sid in stale:
                del self._session_locks[sid]
            if stale:
                logger.info("Cleaned up {} stale session locks", len(stale))
        await self.session_manager.cleanup_stale_load_locks()

    async def _emit_error_and_done(
        self,
        *,
        session_id: str,
        error_message: str,
        code: str | None = None,
        hint: str | None = None,
    ) -> None:
        error_data: dict[str, object] = {"error": error_message}
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
        context = None
        try:
            context = await self.session_manager.get_session(session_id)
            if context.current_message:
                message_id = context.current_message.message_id
        except Exception as exc:
            logger.warning("Failed to retrieve session context for error event: {}", exc)

        await self._emit_event(
            Event(
                type=EventType.MESSAGE_DONE,
                session_id=session_id,
                data={
                    "message_id": message_id,
                    "tokens": TokenUsage(),
                    "context_budget": context.context_budget if context else ContextBudget(),
                    "finish": "error",
                },
            )
        )

    async def _handle_user_message(self, event: Event) -> None:
        session_id = event.session_id
        logger.info("Handling user message for session {}", session_id)
        lock = await self._get_session_lock(session_id)

        async with lock:
            try:
                context = await self.session_manager.get_session(session_id)

                # E-01: 从 ERROR 状态自动恢复到 IDLE，允许重新处理
                if context.state == SessionState.ERROR:
                    await self.session_manager.update_state(session_id, SessionState.IDLE)

                if not context.messages or context.messages[-1].role != "user":
                    raise ValueError(
                        f"No user message found in context for session {session_id}. Please call context.build_user_message() first."
                    )

                task = asyncio.current_task()
                if task is not None:
                    self._active_tasks[session_id] = task

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
                error_message, code, hint = LLMTurnHandler.resolve_error_payload(exc)
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

            finally:
                _ = self._active_tasks.pop(session_id, None)
                await self._try_cleanup_session_lock(session_id)

    async def _conversation_loop(self, session_id: str) -> None:
        context = await self.session_manager.get_session(session_id)
        await self.session_manager.update_state(session_id, SessionState.PROCESSING)

        logger.info("Starting conversation loop for session {}", session_id)

        entered_error_path = False
        try:
            iteration_count = 0
            all_compaction_events: list[CompactionPayload] = []

            while iteration_count < self.max_iterations:
                iteration_count += 1
                result = await self._llm_turn_handler.run(session_id)

                all_compaction_events.extend(result.context_compaction_events)

                if result.tool_calls and result.finish_reason == "tool_calls":
                    tool_calls = result.tool_calls
                    logger.info("LLM requested {} tool calls", len(tool_calls))

                    await self.session_manager.update_state(session_id, SessionState.WAITING_TOOL)
                    _ = await self._execute_tools(session_id, tool_calls)
                    await self.session_manager.update_state(session_id, SessionState.PROCESSING)
                    continue

                logger.info(
                    "Conversation completed for session {} (finish_reason={})",
                    session_id,
                    result.finish_reason,
                )

                await self.session_manager.finish_assistant_message(
                    session_id=session_id,
                    tokens=result.tokens,
                    finish=result.finish_reason or "stop",
                )

                await self._emit_event(
                    Event(
                        type=EventType.MESSAGE_DONE,
                        session_id=session_id,
                        data={
                            "message_id": result.message_id,
                            "tokens": result.tokens,
                            "context_budget": result.context_budget,
                            "context_compaction": result.context_compaction,
                            "context_compaction_events": all_compaction_events,
                            "finish": result.finish_reason or "stop",
                        },
                    )
                )
                return

            logger.error(
                "Max iterations exceeded for session {}: limit={}",
                session_id,
                self.max_iterations,
            )
            # 关闭未完成的 assistant message，避免存储中出现残缺消息
            last_message_id = (
                context.current_message.message_id if context.current_message else None
            )
            if context.current_message:
                await self.session_manager.finish_assistant_message(
                    session_id=session_id,
                    tokens=TokenUsage(),
                    finish="max_iterations_exceeded",
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
                        "message_id": last_message_id,
                        "tokens": TokenUsage(),
                        "context_budget": context.context_budget,
                        "finish": "max_iterations_exceeded",
                    },
                )
            )

        except Exception:
            entered_error_path = True
            await self.session_manager.update_state(session_id, SessionState.ERROR)
            # 关闭未完成的 assistant message，避免下次复用半成品消息
            try:
                err_ctx = await self.session_manager.get_session(session_id)
                if err_ctx.current_message:
                    await self.session_manager.finish_assistant_message(
                        session_id=session_id,
                        tokens=TokenUsage(),
                        finish="error",
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to finalize assistant message on error for session {}: {}",
                    session_id,
                    exc,
                )
            raise

        finally:
            if not entered_error_path:
                try:
                    await self.session_manager.update_state(session_id, SessionState.IDLE)
                except Exception as exc:
                    logger.warning("Failed to reset session {} state to IDLE: {}", session_id, exc)

    async def _execute_tools(
        self, session_id: str, tool_calls: list[ToolCallPayload]
    ) -> list[ToolResult]:
        context = await self.session_manager.get_session(session_id)
        message_id = context.current_message.message_id if context.current_message else None

        if not message_id:
            logger.warning("No current message for tool calls")
            return []

        agent = self.framework.get_agent(context.agent_name)
        if not agent:
            raise RuntimeError(
                f"Agent '{context.agent_name}' not found. Cannot execute tool calls without a registered agent."
            )

        tool_parts_map: dict[str, ToolPart] = {}
        for tool_call in tool_calls:
            tool_part = ToolPart.create_running(session_id, message_id, tool_call)
            tool_parts_map[tool_call.id] = tool_part
            context.add_part_to_current(tool_part)

            await self._emit_event(
                Event(
                    type=EventType.PART_CREATED,
                    session_id=session_id,
                    data=tool_part.model_dump(),
                )
            )

        results = await agent.tool_executor.execute_batch(tool_calls)
        tool_args_by_call_id: dict[str, dict[str, object]] = {}
        for tool_call in tool_calls:
            raw_args = tool_call.function.arguments or "{}"
            try:
                loaded: object = json.loads(raw_args)  # pyright: ignore[reportAny]
                parsed: dict[str, object] = (
                    cast(dict[str, object], loaded)
                    if isinstance(loaded, dict)
                    else {"raw": raw_args}
                )
            except Exception:
                parsed = {"raw": raw_args}
            tool_args_by_call_id[tool_call.id] = parsed

        for result in results:
            call_id = result.tool_call_id
            if call_id not in tool_parts_map:
                continue

            original_part = tool_parts_map[call_id]
            if result.success:
                processed_result = await agent.context.strategy_manager.run_postprocess(
                    agent_name=context.agent_name,
                    session_id=session_id,
                    tool_result=result,
                    tool_arguments=tool_args_by_call_id.get(call_id, {}),
                    storage=self.session_manager.storage,
                )
                updated_part = original_part.update_to_completed(processed_result)
            else:
                updated_part = original_part.update_to_error(result)

            if context.current_message is not None:
                for i, part in enumerate(context.current_message.parts):
                    if isinstance(part, ToolPart) and part.call_id == call_id:
                        context.current_message.parts[i] = updated_part
                        break

            await self._emit_event(
                Event(
                    type=EventType.PART_CREATED,
                    session_id=session_id,
                    data=updated_part.model_dump(),
                )
            )

        return results
