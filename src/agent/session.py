"""会话包装类 - 用户友好的会话接口"""

from typing import Callable, TYPE_CHECKING

from loguru import logger

from .types import SessionState

if TYPE_CHECKING:
    from .core.session import SessionContext, SessionManager


class Session:
    """会话包装类 - 提供监听器管理和数据访问"""

    def __init__(self, context: "SessionContext", session_manager: "SessionManager"):
        self._context = context
        self._session_manager = session_manager
        self.session_id = context.session_id

    def add_listener(self, listener: Callable) -> None:
        """添加监听器"""
        self._session_manager.add_session_listener(self.session_id, listener)
        logger.debug(f"Added listener to session {self.session_id}")

    def remove_listener(self, listener: Callable) -> None:
        """移除监听器"""
        self._session_manager.remove_session_listener(self.session_id, listener)
        logger.debug(f"Removed listener from session {self.session_id}")

    @property
    def state(self) -> SessionState:
        """获取会话状态"""
        return self._context.state

    @property
    def messages(self):
        """获取消息历史"""
        return self._context.messages

    @property
    def agent_name(self) -> str:
        """获取当前 Agent 名称"""
        return self._context.agent_name
