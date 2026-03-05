"""
Agent Context - Agent 运行上下文

这个模块定义了 Agent 运行所需的外部依赖，封装了与 Framework 的交互。
"""

from dataclasses import dataclass
from typing import Union, List

from loguru import logger

from .core.bus import MessageBus
from .core.message import UserMessagePartInput, UserTextPart
from .core.session import SessionManager
from .types import Event, EventType


@dataclass
class AgentContext:
    """
    Agent 运行上下文

    封装 Agent 需要的外部依赖，提供清晰的接口。
    Agent 通过 Context 执行操作，而不是直接依赖 Framework。

    Attributes:
        session_manager: Session 管理器，用于会话操作
        message_bus: 消息总线，用于事件发布
    """

    session_manager: SessionManager
    message_bus: MessageBus

    async def get_session(self, agent_name: str, session_id: str, model_profile_id: str):
        """
        获取或创建会话

        Args:
            agent_name: Agent 名称
            session_id: 会话 ID
            model_profile_id: Framework 注册的模型 profile ID

        Returns:
            Session 对象
        """
        from .session import Session

        context = await self.session_manager.get_or_create_session(
            session_id=session_id,
            model_profile_id=model_profile_id,
            agent_name=agent_name,
        )
        return Session(context, self.session_manager)

    async def send_message(
        self,
        session_id: str,
        message: Union[str, List[UserMessagePartInput]],
    ) -> None:
        """
        发送消息到会话并触发事件

        Args:
            session_id: 会话 ID
            message: 消息内容（字符串或结构化消息）
        """
        # 转换消息格式
        if isinstance(message, str):
            parts = [UserTextPart(text=message)]
        else:
            parts = message

        # 添加消息到会话
        await self.session_manager.add_user_message(session_id, parts)

        # 发布消息接收事件
        await self.message_bus.publish(
            Event(
                type=EventType.MESSAGE_RECEIVED,
                session_id=session_id,
                data={},
            )
        )

        logger.debug(f"Message sent to session {session_id}")

    async def switch_session_agent(self, session_id: str, new_agent_name: str) -> None:
        """
        切换会话的 Agent

        Args:
            session_id: 会话 ID
            new_agent_name: 新 Agent 名称
        """
        await self.session_manager.switch_session_agent(session_id, new_agent_name)
        logger.info(f"Session {session_id} switched to agent {new_agent_name}")
