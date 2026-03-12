"""会话管理器 - 会话生命周期和持久化管理"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import cast

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

        self._session_listeners: dict[str, list[Callable[[Event], Awaitable[None]]]] = {}

        logger.info(f"SessionManager initialized with {self.storage.__class__.__name__}")

    async def _load_session_into_cache(
        self,
        session_id: str,
        session_data: SessionMetadataData,
    ) -> SessionContext:
        """将存储中的会话数据反序列化并加入缓存（调用方需持有 _cache_lock）"""
        context = SessionContext.from_snapshot(session_data)

        messages = await self.storage.load_messages(
            session_id, from_message_id=context.last_compaction_message_id
        )
        context.messages.extend(messages)
        logger.debug(f"Loaded {len(messages)} messages for session {session_id}")

        state_machine = SessionStateMachine(
            session_id=session_id,
            on_state_change=self._on_state_change,
        )

        self._cache[session_id] = context
        self._state_machines[session_id] = state_machine

        return context

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

        # 加入缓存（双重检查）
        async with self._cache_lock:
            # 如果已在缓存中，说明其他协程已创建，直接返回已有的
            if session_id in self._cache:
                logger.warning(
                    f"Session {session_id} already in cache during creation, using existing"
                )
                return self._cache[session_id]

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

        logger.info(f"Created session: {session_id}")
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

        # 在锁内检查是否已存在，避免 TOCTOU 竞态
        async with self._cache_lock:
            if session_id in self._cache:
                raise ValueError(f"Session {session_id} already exists in cache")

            if await self.storage.session_exists(session_id):
                raise ValueError(f"Session {session_id} already exists in storage")

        try:
            return await self._create_and_save_session(session_id, model_profile_id, agent_name)

        except Exception as e:
            # 回滚：如果保存失败，确保缓存中也没有
            async with self._cache_lock:
                if session_id in self._cache:
                    del self._cache[session_id]
                if session_id in self._state_machines:
                    del self._state_machines[session_id]

            # 尝试清理存储
            try:
                await self.storage.delete_session(session_id)
            except Exception:
                pass

            logger.error(f"Failed to create session {session_id}: {e}")
            raise

    async def get_session(self, session_id: str) -> SessionContext:
        """获取会话（带缓存和延迟加载）"""
        # 1. 快速路径：先不加锁检查缓存（避免锁竞争）
        cached = self._cache.get(session_id)
        if cached is not None:
            return cached

        # 2. 慢速路径：需要从存储加载，使用锁保护
        async with self._cache_lock:
            # 双重检查：可能在等待锁期间其他协程已经加载了
            if session_id in self._cache:
                return self._cache[session_id]

            # 3. 从存储加载
            session_data = await self.storage.load_session(session_id)

            if not session_data:
                raise ValueError(f"Session {session_id} not found")

            # 4. 反序列化、加载消息、加入缓存
            context = await self._load_session_into_cache(session_id, session_data)

            logger.info(f"Loaded session from storage: {session_id}")
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
        # 1. 快速路径：检查缓存
        cached = self._cache.get(session_id)
        if cached is not None:
            logger.debug(f"Session {session_id} found in cache")
            await self._refresh_model_profile_if_needed(session_id, cached, model_profile_id)
            return cached

        # 2. 慢速路径：需要锁保护（避免竞态条件）
        async with self._cache_lock:
            # 双重检查：可能在等待锁期间其他协程已经加载或创建了
            if session_id in self._cache:
                context = self._cache[session_id]
                await self._refresh_model_profile_if_needed(session_id, context, model_profile_id)
                return context

            # 3. 尝试从 Storage 加载（在锁内直接加载，避免调用 get_session）
            if await self.storage.session_exists(session_id):
                session_data = await self.storage.load_session(session_id)
                if session_data:
                    # 反序列化、加载消息、加入缓存
                    context = await self._load_session_into_cache(session_id, session_data)

                    await self._refresh_model_profile_if_needed(
                        session_id, context, model_profile_id
                    )

                    logger.info(f"Loaded session from storage: {session_id}")
                    return context

        # 在锁外创建新会话
        try:
            logger.info(f"Creating new session {session_id}")
            return await self._create_and_save_session(session_id, model_profile_id, agent_name)

        except Exception as e:
            # 回滚
            async with self._cache_lock:
                if session_id in self._cache:
                    del self._cache[session_id]
                if session_id in self._state_machines:
                    del self._state_machines[session_id]

            logger.error(f"Failed to save session {session_id}: {e}")
            raise

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
        logger.info(f"Session {session_id} model_profile_id refreshed: {model_profile_id}")

    async def delete_session(self, session_id: str) -> None:
        """删除会话"""
        # 1. 在锁内从缓存删除
        async with self._cache_lock:
            if session_id in self._cache:
                # 终止状态机
                if session_id in self._state_machines:
                    machine = self._state_machines[session_id]
                    trigger = cast("Callable[[], Awaitable[None]]", getattr(machine, "terminate"))
                    await trigger()
                    del self._state_machines[session_id]

                del self._cache[session_id]

        # 2. 存储操作在锁外执行（避免阻塞其他操作）
        await self.storage.delete_session(session_id)

        # 3. 发送终止事件（在锁外执行）
        await self.bus.publish(
            Event(
                type=EventType.SESSION_TERMINATED,
                session_id=session_id,
            )
        )

        logger.info(f"Deleted session: {session_id}")

    async def add_user_message(self, session_id: str, parts: list[UserInput]) -> Message:
        """添加用户消息并自动保存"""
        context = await self.get_session(session_id)
        message = context.build_user_message(parts)

        # 自动保存消息
        await self.storage.save_message(session_id, message)

        logger.debug(f"Added and saved user message: {message.info.id}")
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

        logger.debug(f"Started assistant message: {message.info.id}")
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
            logger.debug(f"Finished and saved assistant message: {finished_message.info.id}")

    async def save_session_metadata(self, session_id: str) -> None:
        """回写会话元数据（不涉及消息历史）。"""
        context = await self.get_session(session_id)
        await self.storage.save_session(session_id, context.metadata)

    async def add_part_to_current(self, session_id: str, part: Part) -> None:
        """添加 Part 到当前消息"""
        context = await self.get_session(session_id)
        context.add_part_to_current(part)

        logger.debug(f"Added part to current message: {part.__class__.__name__}")

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

        logger.info(f"Session {session_id} agent switched: {old_agent} -> {agent_name}")

    async def update_state(self, session_id: str, new_state: SessionState) -> None:
        """更新会话状态"""
        context = await self.get_session(session_id)

        # 获取状态机
        async with self._cache_lock:
            state_machine = self._state_machines.get(session_id)
            if not state_machine:
                logger.warning(f"State machine for session {session_id} not found")
                return

        # 状态转换
        success = await state_machine.transition_to(new_state)

        if success:
            context.update_state(new_state)
            logger.debug(f"Session {session_id} state updated to {new_state.value}")
        else:
            logger.warning(f"Failed to transition session {session_id} to {new_state.value}")

    def add_session_listener(
        self, session_id: str, listener: Callable[[Event], Awaitable[None]]
    ) -> None:
        """添加会话监听器"""
        if session_id not in self._session_listeners:
            self._session_listeners[session_id] = []
        self._session_listeners[session_id].append(listener)
        logger.debug(f"Added listener to session {session_id}")

    def remove_session_listener(
        self, session_id: str, listener: Callable[[Event], Awaitable[None]]
    ) -> None:
        """移除会话监听器"""
        if session_id in self._session_listeners:
            try:
                self._session_listeners[session_id].remove(listener)
                logger.debug(f"Removed listener from session {session_id}")
                # 如果没有监听器了，删除该会话的监听器列表
                if not self._session_listeners[session_id]:
                    del self._session_listeners[session_id]
            except ValueError:
                logger.warning(f"Listener not found in session {session_id}")

    async def emit_to_session_listeners(self, session_id: str, event: Event) -> None:
        """发送事件到会话监听器"""
        listeners = list(self._session_listeners.get(session_id, []))
        if not listeners:
            return

        # 并发调用所有监听器
        results = await asyncio.gather(
            *[listener(event) for listener in listeners], return_exceptions=True
        )

        # 记录错误
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Session listener {i} failed for session {session_id}: {result}",
                    exc_info=result,
                )

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
