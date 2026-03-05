"""
Session Context - 会话上下文管理
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger

from agent.types import (
    ContextBudget,
    ContextBudgetSource,
    Message as ChatMessage,
    SessionData,
    SessionMetadataData,
    SessionState,
)
from agent.constants import UUID_PREFIX_LENGTH
from .context_budget_utils import (
    update_context_budget_from_raw,
    update_context_budget_values,
)
from agent.core.message import (
    Message,
    UserMessageInfo,
    AssistantMessageInfo,
    Part,
    UserMessagePartInput,
    create_part_from_user_input,
)


class SessionContext:
    """
    会话上下文 - 会话数据容器和消息构建器

    职责：
    1. 存储会话状态和配置
    2. 维护消息列表
    3. 提供消息构建方法（注意：不负责持久化）

    注意：
    - 消息构建后会自动加入 messages 列表
    - 持久化由 SessionManager 负责
    - 不应该直接修改 messages 列表，使用提供的方法
    """

    def __init__(
        self,
        session_id: str,
        model_profile_id: str,
        agent_name: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id
        self.state = SessionState.IDLE
        self.model_profile_id = model_profile_id
        self.agent_name = agent_name
        self.metadata = metadata or {}
        self.context_budget: ContextBudget = ContextBudget()
        self.messages: List[Message] = []
        self.current_message: Optional[Message] = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def build_user_message(self, parts: List[UserMessagePartInput]) -> Message:
        message_id = f"msg_{uuid4().hex[:UUID_PREFIX_LENGTH]}"
        message = Message(
            info=UserMessageInfo(
                id=message_id,
                session_id=self.session_id,
                time={"created": int(time.time() * 1000)},
                agent=self.agent_name,
            ),
            parts=[
                create_part_from_user_input(part_data, self.session_id, message_id)
                for part_data in parts
            ],
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message

    def build_assistant_message(
        self,
        parent_id: str,
        *,
        provider_id: str,
        model_id: str,
    ) -> Message:
        message_id = f"msg_{uuid4().hex[:UUID_PREFIX_LENGTH]}"
        message = Message(
            info=AssistantMessageInfo(
                id=message_id,
                session_id=self.session_id,
                parent_id=parent_id,
                time={"created": int(time.time() * 1000)},
                agent=self.agent_name,
                model_id=model_id,
                provider_id=provider_id,
            ),
            parts=[],
        )
        self.messages.append(message)
        self.current_message = message
        self.updated_at = datetime.now()
        return message

    def add_part_to_current(self, part: Part) -> None:
        if self.current_message:
            self.current_message.add_part(part)
            self.updated_at = datetime.now()

    def finish_current_message(
        self,
        cost: float = 0.0,
        tokens: Optional[Dict[str, Any]] = None,
        finish: str = "stop",
    ) -> None:
        if self.current_message and isinstance(self.current_message.info, AssistantMessageInfo):
            self.current_message.info.time["completed"] = int(time.time() * 1000)
            self.current_message.info.cost = cost
            self.current_message.info.tokens = tokens or {}
            self.current_message.info.finish = finish
            self.current_message = None
            self.updated_at = datetime.now()

    def update_state(self, new_state: SessionState) -> None:
        self.state = new_state
        self.updated_at = datetime.now()

    def switch_agent(self, new_agent_name: str) -> None:
        old_agent = self.agent_name
        self.agent_name = new_agent_name
        self.updated_at = datetime.now()
        logger.info(f"Session {self.session_id} switched agent: {old_agent} -> {new_agent_name}")

    def update_model_profile(self, model_profile_id: Optional[str]) -> None:
        """更新会话绑定的模型 profile。"""
        self.model_profile_id = model_profile_id
        self.updated_at = datetime.now()

    def update_context_budget(
        self,
        *,
        total_tokens: int | None,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        source: ContextBudgetSource,
    ) -> ContextBudget:
        self.context_budget = update_context_budget_values(
            self.context_budget,
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            source=source,
        )
        self.updated_at = datetime.now()
        return self.context_budget

    def get_llm_messages(self) -> List[ChatMessage]:
        llm_messages: List[ChatMessage] = []
        for message in self.messages:
            llm_messages.extend(message.to_llm_messages())
        return llm_messages

    def to_metadata_dict(self) -> SessionMetadataData:
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "model_profile_id": self.model_profile_id,
            "agent_name": self.agent_name,
            "metadata": self.metadata,
            "context_budget": self.context_budget.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def to_dict(self) -> SessionData:
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "model_profile_id": self.model_profile_id,
            "agent_name": self.agent_name,
            "metadata": self.metadata,
            "context_budget": self.context_budget.to_dict(),
            "message_count": len(self.messages),
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionContext":
        context = cls(
            session_id=data["session_id"],
            model_profile_id=data.get("model_profile_id") or "",
            agent_name=data.get("agent_name", "general"),
            metadata=data.get("metadata", {}),
        )
        update_context_budget_from_raw(context.context_budget, data.get("context_budget"))
        if "state" in data:
            context.state = SessionState(data["state"])
        if "created_at" in data:
            context.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            context.updated_at = datetime.fromisoformat(data["updated_at"])
        for msg_data in data.get("messages", []):
            message = Message.from_dict(msg_data)
            context.messages.append(message)
        return context
