"""
Agent Event Loop - Agent事件循环
外层事件驱动（支持多session并发），内层委托 ConversationRunner 处理对话循环。
"""

from __future__ import annotations

import asyncio

from loguru import logger

from agent.core.bus import MessageBus
from agent.core.loop.conversation_runner import ConversationRunner
from agent.core.loop.llm_turn_handler import LLMTurnHandler
from agent.core.session import SessionManager
from agent.core.session.types import ContextBudget
from agent.types import (
    Event,
    EventType,
    SessionState,
    TokenUsage,
)

from agent.core.loop._framework_protocol import FrameworkProtocol


class AgentEventLoop:
    """Agent事件循环。外层事件驱动，内层委托 ConversationRunner。"""

    def __init__(
        self,
        bus: MessageBus,
        session_manager: SessionManager,
        adapter_registry: object,  # 保留参数兼容性，不再使用
        framework: FrameworkProtocol,
        max_iterations: int = 100,
    ):
        self.bus: MessageBus = bus
        self.session_manager: SessionManager = session_manager
        self.framework: FrameworkProtocol = framework
        self.max_iterations: int = max(1, max_iterations)
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._session_locks_guard: asyncio.Lock = asyncio.Lock()
        self._active_tasks: dict[str, asyncio.Task[None]] = {}

        self._llm_turn_handler: LLMTurnHandler = LLMTurnHandler(
            emit_event=self._emit_event,
        )

        self.conversation_runner: ConversationRunner = ConversationRunner(
            turn_handler=self._llm_turn_handler,
            emit_event=self._emit_event,
            max_iterations=max_iterations,
        )

        self.bus.subscribe(EventType.MESSAGE_RECEIVED, self._handle_user_message)
        self.bus.subscribe(EventType.SESSION_TERMINATED, self._handle_session_terminated)
        self.bus.subscribe(EventType.SESSION_CANCELLED, self._handle_session_cancelled)
        logger.info("AgentEventLoop initialized (max_iterations={})", self.max_iterations)

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

    async def _handle_session_cancelled(self, event: Event) -> None:
        session_id = event.session_id
        task = self._active_tasks.pop(session_id, None)
        if task is not None and not task.done():
            _ = task.cancel()
            logger.info("Cancelled active task for session {} (non-destructive)", session_id)

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
        async with self._session_locks_guard:
            lock = self._session_locks.get(session_id)
            if lock is not None and not lock.locked():
                del self._session_locks[session_id]

    async def cleanup_stale_locks(self) -> None:
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
            Event(type=EventType.LLM_ERROR, session_id=session_id, data=error_data)
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

                if context.state == SessionState.ERROR:
                    await self.session_manager.update_state(session_id, SessionState.IDLE)

                if not context.messages or context.messages[-1].role != "user":
                    raise ValueError(f"No user message found in context for session {session_id}.")

                # 从 framework 找到 agent，传给 ConversationRunner
                agent = self.framework.get_agent(context.agent_name)
                if agent is None:
                    raise ValueError(f"Agent '{context.agent_name}' not found")

                task = asyncio.current_task()
                if task is not None:
                    self._active_tasks[session_id] = task

                _ = await self.conversation_runner.conversation_loop(session_id, agent)

            except ValueError as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                logger.error("Error handling user message (invalid state): {}", error_message)
                await self._emit_error_and_done(
                    session_id=session_id, error_message=error_message, code="INVALID_STATE"
                )

            except Exception as exc:
                error_message, code, hint = LLMTurnHandler.resolve_error_payload(exc)
                logger.exception(
                    "Error handling user message for session {}: {}", session_id, error_message
                )
                await self._emit_error_and_done(
                    session_id=session_id, error_message=error_message, code=code, hint=hint
                )

            finally:
                _ = self._active_tasks.pop(session_id, None)
                await self._try_cleanup_session_lock(session_id)
