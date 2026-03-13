"""会话管理器 - 会话生命周期和持久化管理"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable

from loguru import logger

from agent.core.bus import MessageBus
from agent.core.message import Message, Part, UserInput
from agent.types import Event, EventType, SessionMetadataData, SessionState, TokenUsage
from agent.core.storage import StorageProvider, MemoryStorage
from .context import SessionContext
from .state import SessionStateMachine


class SessionManager:
    """
    会话管理器 - 管理所有会话的生命周期

    职责：
    1. 会话生命周期管理（创建、获取、删除）
    2. 缓存管理（内存缓存 + 延迟加载）
    3. 持久化管理（自动保存到 Storage）
    4. 消息管理（创建消息 + 自动保存）
    5. 状态管理（状态转换 + 状态机）
    6. 监听器管理（会话级事件监听）

    架构：
    - 内存缓存：快速访问
    - Storage：持久化层
    - 并发安全：使用 Lock 保护
    """

    def __init__(
        self,
        bus: MessageBus,
        storage: StorageProvider | None = None,
    ):
        self.bus: MessageBus = bus
        self.storage: StorageProvider = storage or MemoryStorage()

        self._cache: dict[str, SessionContext] = {}
        self._state_machines: dict[str, SessionStateMachine] = {}

        self._cache_lock: asyncio.Lock = asyncio.Lock()
        self._session_load_locks: dict[str, asyncio.Lock] = {}

        self._session_listeners: dict[str, list[Callable[[Event], Awaitable[None]]]] = {}
        self._listener_tasks: dict[str, asyncio.Task[None]] = {}
        self._listener_tasks_lock: asyncio.Lock = asyncio.Lock()

        logger.info("SessionManager initialized with {}", self.storage.__class__.__name__)

    async def _load_session_from_storage(
        self,
        session_id: str,
        session_data: SessionMetadataData,
    ) -> SessionContext:
        """从存储中反序列化会话数据和消息（不访问缓存，可在锁外调用）"""
        context = SessionContext.from_snapshot(session_data)
        messages = await self.storage.load_messages(
            session_id, from_message_id=context.last_compaction_message_id
        )
        context.messages.extend(messages)
        logger.debug("Loaded {} messages for session {}", len(messages), session_id)
        return context

    def _put_session_in_cache(self, session_id: str, context: SessionContext) -> None:
        """将会话写入缓存（调用方需持有 _cache_lock）"""
        state_machine = SessionStateMachine(
            session_id=session_id,
            on_state_change=self._on_state_change,
        )
        self._cache[session_id] = context
        self._state_machines[session_id] = state_machine

    async def _create_and_save_session(
        self,
        session_id: str,
        model_profile_id: str,
        agent_name: str,
    ) -> SessionContext:
        """创建会话、保存到存储、加入缓存、发布事件（内部方法）

        注意：调用前应确保 session 不存在，否则可能失败
        """
        # 创建会话上下文
        context = SessionContext(
            session_id=session_id,
            model_profile_id=model_profile_id,
            agent_name=agent_name,
        )

        # 创建状态机
        state_machine = SessionStateMachine(
            session_id=session_id,
            on_state_change=self._on_state_change,
        )

        # 保存到存储
        await self.storage.save_session(session_id, context.metadata)

        # 加入缓存（调用方应持有 _cache_lock）
        self._cache[session_id] = context
        self._state_machines[session_id] = state_machine

        # 发布事件
        await self.bus.publish(
            Event(
                type=EventType.SESSION_CREATED,
                session_id=session_id,
                data=context.snapshot.model_dump(),
            )
        )

        logger.info("Created session: {}", session_id)
        return context

    async def create_session(
        self,
        session_id: str | None = None,
        model_profile_id: str = "",
        agent_name: str = "general",
    ) -> SessionContext:
        """创建新会话"""
        if not model_profile_id:
            raise ValueError("model_profile_id is required")
        if not session_id:
            session_id = str(uuid.uuid4())

        # 在锁内检查并创建，避免 TOCTOU 竞态
        async with self._cache_lock:
            if session_id in self._cache:
                raise ValueError(f"Session {session_id} already exists in cache")

            if await self.storage.session_exists(session_id):
                raise ValueError(f"Session {session_id} already exists in storage")

            try:
                return await self._create_and_save_session(session_id, model_profile_id, agent_name)

            except Exception as e:
                # 回滚：如果保存失败，确保缓存中也没有
                if session_id in self._cache:
                    del self._cache[session_id]
                if session_id in self._state_machines:
                    del self._state_machines[session_id]

                # 尝试清理存储
                try:
                    await self.storage.delete_session(session_id)
                except Exception:
                    logger.warning("Failed to cleanup storage for session {}", session_id)

                logger.error("Failed to create session {}: {}", session_id, e)
                raise

    def _get_session_load_lock(self, session_id: str) -> asyncio.Lock:
        """获取或创建 per-session 加载锁。"""
        lock = self._session_load_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_load_locks[session_id] = lock
        return lock

    async def _cleanup_session_load_lock(self, session_id: str) -> None:
        """清理不再需要的 per-session 加载锁。

        在 session 成功加载到缓存后调用。如果 lock 当前没被占用，
        将其从 _session_load_locks 中移除以防止无界增长。
        必须在 _cache_lock 内执行以保证安全。
        """
        async with self._cache_lock:
            lock = self._session_load_locks.get(session_id)
            if lock is not None and not lock.locked():
                del self._session_load_locks[session_id]

    async def get_session(self, session_id: str) -> SessionContext:
        """获取会话（带缓存和延迟加载）

        使用两阶段加锁：全局锁仅用于缓存查找，per-session 锁保护存储加载，
        避免全局锁长期持有阻塞其他会话。
        """
        # 快速路径：缓存命中
        async with self._cache_lock:
            cached = self._cache.get(session_id)
            if cached is not None:
                return cached
            load_lock = self._get_session_load_lock(session_id)

        # 慢路径：per-session 锁保护存储加载
        async with load_lock:
            # double-check: 另一个协程可能已完成加载
            async with self._cache_lock:
                cached = self._cache.get(session_id)
                if cached is not None:
                    return cached

            # 在锁外做存储 I/O
            session_data = await self.storage.load_session(session_id)
            if not session_data:
                raise ValueError(f"Session {session_id} not found")

            # 在锁外做反序列化和消息加载
            context = await self._load_session_from_storage(session_id, session_data)

            # 写入缓存
            async with self._cache_lock:
                # 再次 double-check
                cached = self._cache.get(session_id)
                if cached is not None:
                    return cached
                self._put_session_in_cache(session_id, context)

            logger.info("Loaded session from storage: {}", session_id)

        await self._cleanup_session_load_lock(session_id)
        return context

    async def get_or_create_session(
        self,
        session_id: str,
        model_profile_id: str,
        agent_name: str = "general",
    ) -> SessionContext:
        """获取或创建会话（按需创建）"""
        if not model_profile_id:
            raise ValueError("model_profile_id is required")

        # 快速路径：缓存命中
        async with self._cache_lock:
            cached = self._cache.get(session_id)
            if cached is not None:
                logger.debug("Session {} found in cache", session_id)
                await self._refresh_model_profile_if_needed(session_id, cached, model_profile_id)
                return cached
            load_lock = self._get_session_load_lock(session_id)

        # 慢路径：per-session 锁
        async with load_lock:
            async with self._cache_lock:
                cached = self._cache.get(session_id)
                if cached is not None:
                    await self._refresh_model_profile_if_needed(
                        session_id, cached, model_profile_id
                    )
                    return cached

            # 锁外做存储 I/O
            if await self.storage.session_exists(session_id):
                session_data = await self.storage.load_session(session_id)
                if session_data:
                    context = await self._load_session_from_storage(session_id, session_data)
                    async with self._cache_lock:
                        # double-check
                        cached = self._cache.get(session_id)
                        if cached is not None:
                            await self._refresh_model_profile_if_needed(
                                session_id, cached, model_profile_id
                            )
                            return cached
                        self._put_session_in_cache(session_id, context)

                    await self._refresh_model_profile_if_needed(
                        session_id, context, model_profile_id
                    )
                    logger.info("Loaded session from storage: {}", session_id)
                    await self._cleanup_session_load_lock(session_id)
                    return context

            # 创建新会话
            async with self._cache_lock:
                # 再次 double-check（可能在等锁期间被创建）
                cached = self._cache.get(session_id)
                if cached is not None:
                    await self._refresh_model_profile_if_needed(
                        session_id, cached, model_profile_id
                    )
                    return cached

                try:
                    logger.info("Creating new session {}", session_id)
                    context = await self._create_and_save_session(
                        session_id, model_profile_id, agent_name
                    )
                except Exception as e:
                    if session_id in self._cache:
                        del self._cache[session_id]
                    if session_id in self._state_machines:
                        del self._state_machines[session_id]
                    logger.error("Failed to save session {}: {}", session_id, e)
                    raise

        await self._cleanup_session_load_lock(session_id)
        return context

    async def _refresh_model_profile_if_needed(
        self,
        session_id: str,
        context: SessionContext,
        model_profile_id: str,
    ) -> None:
        """刷新已有会话绑定的 model_profile_id 并回写存储。"""
        if context.model_profile_id == model_profile_id:
            return

        context.update_model_profile(model_profile_id)
        await self.storage.save_session(session_id, context.metadata)
        logger.info("Session {} model_profile_id refreshed: {}", session_id, model_profile_id)

    async def cached_session_ids(self) -> list[str]:
        """Return IDs of sessions currently in the in-memory cache."""
        async with self._cache_lock:
            return list(self._cache.keys())

    async def cleanup_stale_load_locks(self) -> None:
        """Remove per-session load locks for sessions no longer in cache."""
        async with self._cache_lock:
            stale = [sid for sid in self._session_load_locks if sid not in self._cache]
            for sid in stale:
                del self._session_load_locks[sid]
            if stale:
                logger.debug("Cleaned up {} stale session load locks", len(stale))

    async def delete_session(self, session_id: str) -> None:
        """删除会话"""
        # 1. 在锁内从缓存删除
        async with self._cache_lock:
            if session_id in self._cache:
                # 终止状态机
                if session_id in self._state_machines:
                    machine = self._state_machines[session_id]
                    _ = await machine.transition_to(SessionState.TERMINATED)
                    del self._state_machines[session_id]

                del self._cache[session_id]
            _ = self._session_load_locks.pop(session_id, None)

        # 2. 存储操作在锁外执行（避免阻塞其他操作）
        await self.storage.delete_session(session_id)

        async with self._listener_tasks_lock:
            listener_task = self._listener_tasks.pop(session_id, None)
        if listener_task and not listener_task.done():
            _ = listener_task.cancel()
            try:
                await listener_task
            except asyncio.CancelledError:
                pass

        # 3. 发送终止事件（在锁外执行）
        await self.bus.publish(
            Event(
                type=EventType.SESSION_TERMINATED,
                session_id=session_id,
            )
        )

        logger.info("Deleted session: {}", session_id)

    async def add_user_message(self, session_id: str, parts: list[UserInput]) -> Message:
        """添加用户消息并自动保存"""
        context = await self.get_session(session_id)
        message = context.build_user_message(parts)

        # 自动保存消息
        await self.storage.save_message(session_id, message)

        logger.debug("Added and saved user message: {}", message.info.id)
        return message

    async def start_assistant_message(
        self,
        session_id: str,
        parent_id: str,
        *,
        provider_id: str,
        model_id: str,
    ) -> Message:
        """开始 AI 助手消息（不保存，等完成后再保存）"""
        context = await self.get_session(session_id)
        message = context.build_assistant_message(
            parent_id, provider_id=provider_id, model_id=model_id
        )

        logger.debug("Started assistant message: {}", message.info.id)
        return message

    async def finish_assistant_message(
        self,
        session_id: str,
        tokens: TokenUsage | None = None,
        finish: str = "stop",
    ) -> None:
        """
        完成 AI 助手消息并自动保存

        Args:
            session_id: 会话ID
            tokens: Token 统计
            finish: 结束原因
        """
        context = await self.get_session(session_id)
        finished_message = context.current_message
        context.finish_current_message(tokens=tokens, finish=finish)

        # 自动保存完成的消息
        if finished_message:
            await self.storage.save_message(session_id, finished_message)
            logger.debug("Finished and saved assistant message: {}", finished_message.info.id)

    async def save_session_metadata(self, session_id: str) -> None:
        """回写会话元数据（不涉及消息历史）。"""
        context = await self.get_session(session_id)
        await self.storage.save_session(session_id, context.metadata)

    async def add_part_to_current(self, session_id: str, part: Part) -> None:
        """添加 Part 到当前消息"""
        context = await self.get_session(session_id)
        context.add_part_to_current(part)

        logger.debug("Added part to current message: {}", part.__class__.__name__)

    async def switch_session_agent(self, session_id: str, agent_name: str) -> None:
        """切换会话的 Agent"""
        context = await self.get_session(session_id)
        old_agent = context.agent_name

        # 切换 Agent
        context.switch_agent(agent_name)

        # 保存到存储（只保存元数据，不包括消息历史）
        await self.storage.save_session(session_id, context.metadata)

        # 发送事件
        await self.bus.publish(
            Event(
                type=EventType.SESSION_AGENT_SWITCHED,
                session_id=session_id,
                data={"old_agent": old_agent, "new_agent": agent_name},
            )
        )

        logger.info("Session {} agent switched: {} -> {}", session_id, old_agent, agent_name)

    async def update_state(self, session_id: str, new_state: SessionState) -> None:
        """更新会话状态"""
        context = await self.get_session(session_id)

        # 获取状态机
        async with self._cache_lock:
            state_machine = self._state_machines.get(session_id)
            if not state_machine:
                logger.warning("State machine for session {} not found", session_id)
                return

        # 状态转换
        success = await state_machine.transition_to(new_state)

        if success:
            context.update_state(new_state)
            logger.debug("Session {} state updated to {}", session_id, new_state.value)
        else:
            logger.warning("Failed to transition session {} to {}", session_id, new_state.value)

    def add_session_listener(
        self, session_id: str, listener: Callable[[Event], Awaitable[None]]
    ) -> None:
        """添加会话监听器"""
        if session_id not in self._session_listeners:
            self._session_listeners[session_id] = []
        self._session_listeners[session_id].append(listener)
        logger.debug("Added listener to session {}", session_id)

    def remove_session_listener(
        self, session_id: str, listener: Callable[[Event], Awaitable[None]]
    ) -> None:
        """移除会话监听器"""
        if session_id in self._session_listeners:
            try:
                self._session_listeners[session_id].remove(listener)
                logger.debug("Removed listener from session {}", session_id)
                # 如果没有监听器了，删除该会话的监听器列表
                if not self._session_listeners[session_id]:
                    del self._session_listeners[session_id]
            except ValueError:
                logger.warning("Listener not found in session {}", session_id)

    async def emit_to_session_listeners(self, session_id: str, event: Event) -> None:
        """发送事件到会话监听器"""
        listeners = list(self._session_listeners.get(session_id, []))
        if not listeners:
            return

        async def _deliver(previous: asyncio.Task[None] | None) -> None:
            if previous is not None:
                try:
                    await previous
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

            results = await asyncio.gather(
                *[listener(event) for listener in listeners], return_exceptions=True
            )

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        "Session listener {} failed for session {}: {}",
                        i,
                        session_id,
                        result,
                        exc_info=result,
                    )

            # 清理自身引用（在同一 task 内完成，无需 create_task）
            async with self._listener_tasks_lock:
                current = self._listener_tasks.get(session_id)
                if current is asyncio.current_task():
                    _ = self._listener_tasks.pop(session_id, None)

        async with self._listener_tasks_lock:
            previous = self._listener_tasks.get(session_id)
            task = asyncio.create_task(_deliver(previous))
            self._listener_tasks[session_id] = task

    async def _on_state_change(self, session_id: str, from_state: str, to_state: str) -> None:
        """
        状态改变回调

        Args:
            session_id: 会话ID
            from_state: 原状态
            to_state: 新状态
        """
        # 发送状态改变事件
        await self.bus.publish(
            Event(
                type=EventType.SESSION_STATE_CHANGED,
                session_id=session_id,
                data={
                    "from_state": from_state,
                    "to_state": to_state,
                },
            )
        )
