"""
Agent Context - Agent 运行上下文

这个模块定义了 Agent 运行所需的外部依赖，封装了与 Framework 的交互。
"""

from dataclasses import dataclass
from typing import List

from loguru import logger

from .core.bus import MessageBus
from .core.message import UserInput
from .core.runtime import AgentStrategyManager
from .core.session import SessionManager
from .types import Event, EventType, LLMConfig


@dataclass
class AgentContext:
    """
    Agent 运行上下文

    封装 Agent 需要的外部依赖，提供清晰的接口。
    Agent 通过 Context 执行操作，而不是直接依赖 Framework。

    Attributes:
        session_manager: Session 管理器，用于会话操作
        message_bus: 消息总线，用于事件发布
        strategy_manager: 策略管理器，用于运行策略
        agent_name: 当前 Agent 名称
        llm_profiles: Agent 可用的 LLM 配置映射（已根据权限过滤）
    """

    session_manager: SessionManager
    message_bus: MessageBus
    strategy_manager: AgentStrategyManager
    agent_name: str
    llm_profiles: dict[str, LLMConfig]

    @staticmethod
    def _normalize_profile_id(profile_id: str) -> str:
        raw_profile = profile_id.strip()
        if ":" not in raw_profile:
            raise ValueError(
                f"Invalid model profile id '{profile_id}'. Expected format: provider:model"
            )
        provider, model = raw_profile.split(":", 1)
        if not provider.strip() or not model.strip():
            raise ValueError("provider and model are required to build llm profile id")
        return f"{provider.strip().lower()}:{model.strip()}"

    def resolve_llm_config_for_agent(self, agent_name: str, profile_id: str) -> LLMConfig:
        """
        解析 LLM profile

        Args:
            agent_name: Agent 名称
            profile_id: LLM profile id (provider:model)

        Returns:
            LLMConfig
        """
        if agent_name != self.agent_name:
            raise ValueError(
                f"Agent context mismatch: expected '{self.agent_name}', got '{agent_name}'"
            )

        normalized_profile = self._normalize_profile_id(profile_id)

        if normalized_profile not in self.llm_profiles:
            available = ", ".join(sorted(self.llm_profiles.keys())) or "<none>"
            raise ValueError(
                f"LLM profile '{normalized_profile}' not available for agent '{agent_name}'. "
                f"Available: {available}"
            )

        return self.llm_profiles[normalized_profile]

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
        message: List[UserInput],
    ) -> None:
        """
        发送消息到会话并触发事件

        Args:
            session_id: 会话 ID
            message: 消息内容（结构化消息列表）
        """
        # 添加消息到会话
        await self.session_manager.add_user_message(session_id, message)
        
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
