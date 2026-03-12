"""
Session Context - 会话上下文管理
"""

import time
from datetime import datetime, timezone
from loguru import logger

from agent.types import (
    ContextBudget,
    ContextBudgetSource,
    SessionData,
    SessionMetadataData,
    SessionState,
    TokenUsage,
)
from agent.core.utils import gen_id
from agent.core.message import (
    Message,
    UserMessageInfo,
    AssistantMessageInfo,
    CompactionMessageInfo,
    Part,
    UserInput,
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
    ):
        self.session_id: str = session_id
        self.state: SessionState = SessionState.IDLE
        self.model_profile_id: str | None = model_profile_id
        self.agent_name: str = agent_name
        self.context_budget: ContextBudget = ContextBudget()
        self.messages: list[Message] = []
        self.current_message: Message | None = None
        self.last_compaction_message_id: str | None = None
        self.created_at: datetime = datetime.now(tz=timezone.utc)
        self.updated_at: datetime = datetime.now(tz=timezone.utc)

    def build_user_message(self, parts: list[UserInput]) -> Message:
        message_id = gen_id("msg_")
        message = Message(
            info=UserMessageInfo(
                id=message_id,
                session_id=self.session_id,
                created_at=int(time.time() * 1000),
                agent=self.agent_name,
            ),
            parts=[part.to_part(self.session_id, message_id) for part in parts],
        )
        self.messages.append(message)
        self.updated_at = datetime.now(tz=timezone.utc)
        return message

    def build_assistant_message(
        self,
        parent_id: str,
        *,
        provider_id: str,
        model_id: str,
    ) -> Message:
        message_id = gen_id("msg_")
        message = Message(
            info=AssistantMessageInfo(
                id=message_id,
                session_id=self.session_id,
                parent_id=parent_id,
                created_at=int(time.time() * 1000),
                agent=self.agent_name,
                model_id=model_id,
                provider_id=provider_id,
            ),
            parts=[],
        )
        self.messages.append(message)
        self.current_message = message
        self.updated_at = datetime.now(tz=timezone.utc)
        return message

    def get_llm_messages(self) -> list[Message]:
        """返回 LLM 可见的消息视图。

        有 compaction 书签时返回 [书签(含摘要)] + 书签之后的原始消息，否则返回全部。
        书签的 role="compaction" 由 LLMProvider.prepare_messages 映射为 "user"。
        """
        idx = self._find_last_compaction_index()
        if idx is None:
            return list(self.messages)
        return list(self.messages[idx:])

    def _find_last_compaction_index(self) -> int | None:
        for idx in range(len(self.messages) - 1, -1, -1):
            if isinstance(self.messages[idx].info, CompactionMessageInfo):
                return idx
        return None

    def add_part_to_current(self, part: Part) -> None:
        if self.current_message:
            self.current_message.add_part(part)
            self.updated_at = datetime.now(tz=timezone.utc)
        else:
            logger.warning(
                "add_part_to_current called but no current_message exists, part dropped: {}",
                type(part).__name__,
            )

    def finish_current_message(
        self,
        tokens: TokenUsage | None = None,
        finish: str = "stop",
    ) -> None:
        if self.current_message and isinstance(self.current_message.info, AssistantMessageInfo):
            self.current_message.info.completed_at = int(time.time() * 1000)
            self.current_message.info.tokens = tokens or TokenUsage()
            self.current_message.info.finish = finish
            self.current_message = None
            self.updated_at = datetime.now(tz=timezone.utc)

    def update_state(self, new_state: SessionState) -> None:
        self.state = new_state
        self.updated_at = datetime.now(tz=timezone.utc)

    def switch_agent(self, new_agent_name: str) -> None:
        old_agent = self.agent_name
        self.agent_name = new_agent_name
        self.updated_at = datetime.now(tz=timezone.utc)
        logger.info(f"Session {self.session_id} switched agent: {old_agent} -> {new_agent_name}")

    def update_model_profile(self, model_profile_id: str | None) -> None:
        """更新会话绑定的模型 profile。"""
        self.model_profile_id = model_profile_id
        self.updated_at = datetime.now(tz=timezone.utc)

    def update_context_budget(
        self,
        *,
        total_tokens: int | None,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        source: ContextBudgetSource,
    ) -> ContextBudget:
        self.context_budget.update(
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            source=source,
        )
        self.updated_at = datetime.now(tz=timezone.utc)
        return self.context_budget

    @property
    def metadata(self) -> SessionMetadataData:
        """会话元数据快照（不含消息历史）"""
        return SessionMetadataData(
            session_id=self.session_id,
            state=self.state.value,
            model_profile_id=self.model_profile_id or "",
            agent_name=self.agent_name,
            context_budget=self.context_budget,
            last_compaction_message_id=self.last_compaction_message_id,
            created_at=self.created_at.isoformat(),
            updated_at=self.updated_at.isoformat(),
        )

    @property
    def snapshot(self) -> SessionData:
        """完整会话快照（含消息历史）"""
        return SessionData(
            session_id=self.session_id,
            state=self.state.value,
            model_profile_id=self.model_profile_id or "",
            agent_name=self.agent_name,
            context_budget=self.context_budget,
            message_count=len(self.messages),
            messages=[msg.model_dump(mode="json") for msg in self.messages],
            created_at=self.created_at.isoformat(),
            updated_at=self.updated_at.isoformat(),
        )

    @classmethod
    def from_snapshot(cls, data: SessionMetadataData) -> "SessionContext":
        context = cls(
            session_id=data.session_id,
            model_profile_id=data.model_profile_id or "",
            agent_name=data.agent_name,
        )
        context.context_budget = data.context_budget
        context.last_compaction_message_id = data.last_compaction_message_id
        context.state = SessionState(data.state)
        context.created_at = datetime.fromisoformat(data.created_at)
        context.updated_at = datetime.fromisoformat(data.updated_at)

        return context
