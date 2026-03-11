"""
Agent Context - Agent 运行上下文

这个模块定义了 Agent 运行所需的外部依赖，封装了与 Framework 的交互。
"""

from typing import List

from loguru import logger

from .core.bus import MessageBus
from .core.message import UserInput
from .core.utils import normalize_profile_id
from .core.runtime import AgentStrategyManager
from .core.session import SessionManager
from .types import Event, EventType, LLMConfig


class AgentContext:
    """
    Agent 运行上下文

    封装 Agent 需要的外部依赖，提供清晰的接口。
    Agent 通过 Context 执行操作，而不是直接依赖 Framework。
    """

    def __init__(
        self,
        session_manager: SessionManager,
        message_bus: MessageBus,
        strategy_manager: AgentStrategyManager,
        agent_name: str,
        llm_profiles: dict[str, LLMConfig],
    ) -> None:
        self.session_manager = session_manager
        self.message_bus = message_bus
        self.strategy_manager = strategy_manager
        self.agent_name = agent_name
        self.llm_profiles = llm_profiles

    def resolve_llm_config_for_agent(self, agent_name: str, profile_id: str) -> LLMConfig:
        if agent_name != self.agent_name:
            raise ValueError(
                f"Agent context mismatch: expected '{self.agent_name}', got '{agent_name}'"
            )

        normalized_profile = normalize_profile_id(profile_id)

        if normalized_profile not in self.llm_profiles:
            available = ", ".join(sorted(self.llm_profiles.keys())) or "<none>"
            raise ValueError(
                f"LLM profile '{normalized_profile}' not available for agent '{agent_name}'. "
                f"Available: {available}"
            )

        return self.llm_profiles[normalized_profile]

    async def get_session(self, agent_name: str, session_id: str, model_profile_id: str):
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
        await self.session_manager.add_user_message(session_id, message)

        await self.message_bus.publish(
            Event(
                type=EventType.MESSAGE_RECEIVED,
                session_id=session_id,
                data={},
            )
        )

        logger.debug(f"Message sent to session {session_id}")

    async def switch_session_agent(self, session_id: str, new_agent_name: str) -> None:
        await self.session_manager.switch_session_agent(session_id, new_agent_name)
        logger.info(f"Session {session_id} switched to agent {new_agent_name}")
