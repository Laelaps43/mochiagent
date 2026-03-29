"""会话包装类 - 用户友好的会话接口"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import override

from loguru import logger

from .core.message.message import Message
from .core.session.context import SessionContext
from .core.session.manager import SessionManager
from .types import Event, SessionState


class Session:
    """会话包装类 - 提供监听器管理和数据访问"""

    def __init__(self, context: "SessionContext", session_manager: "SessionManager"):
        self._context: SessionContext = context
        self._session_manager: SessionManager = session_manager
        self.session_id: str = context.session_id

    def add_listener(self, listener: "Callable[[Event], Awaitable[None]]") -> None:
        """添加监听器"""
        self._session_manager.add_session_listener(self.session_id, listener)
        logger.debug("Added listener to session {}", self.session_id)

    def remove_listener(self, listener: "Callable[[Event], Awaitable[None]]") -> None:
        """移除监听器"""
        self._session_manager.remove_session_listener(self.session_id, listener)
        logger.debug("Removed listener from session {}", self.session_id)

    @property
    def state(self) -> SessionState:
        """获取会话状态"""
        return self._context.state

    @property
    def messages(self) -> list[Message]:
        """获取消息历史（防御性拷贝）"""
        return list(self._context.messages)

    @property
    def agent_name(self) -> str:
        """获取当前 Agent 名称"""
        return self._context.agent_name

    @property
    def model_profile_id(self) -> str | None:
        """获取会话绑定的模型 profile ID。"""
        return self._context.model_profile_id

    async def cancel(self) -> None:
        """取消 session 执行（非破坏性，级联取消子 session）。"""
        await self._session_manager.cancel_session(self.session_id)

    @override
    def __repr__(self) -> str:
        return (
            f"Session(session_id={self.session_id!r}, "
            f"state={self.state.value!r}, "
            f"agent={self.agent_name!r})"
        )
