"""
Storage Provider - 存储抽象接口
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from agent.core.message import Message
from agent.types import SessionMetadataData


class ArtifactMetadata(BaseModel):
    artifact_ref: str = ""
    artifact_id: str = ""
    session_id: str = ""
    kind: str = ""
    size: int = 0
    path: str = ""
    created_at_ms: int = 0
    metadata: dict[str, object] = Field(default_factory=dict)


class ArtifactReadResult(BaseModel):
    success: bool = False
    error: str = ""
    artifact_ref: str = ""
    path: str = ""
    content: str = ""
    offset: int = 0
    limit: int = 0
    next_offset: int = 0
    eof: bool = False
    size: int = 0


class StorageProvider(ABC):
    """
    存储提供者抽象基类

    用户可以继承此类实现自定义存储后端
    例如：PostgreSQL, MongoDB, Redis, File 等

    设计：
    - 会话元数据和消息分开存储
    - 支持增量保存消息
    - 加载会话时主动加载历史消息
    """

    @abstractmethod
    async def save_session(self, session_id: str, session_data: SessionMetadataData) -> None:
        """
        保存会话元数据（不包括消息列表）

        Args:
            session_id: 会话 ID
            session_data: 会话元数据，包含：
                - session_id
                - state
                - model_profile_id
                - agent_name
                - context_budget
                - created_at
                - updated_at
                注意：不包含 messages 字段
        """
        pass

    @abstractmethod
    async def load_session(self, session_id: str) -> SessionMetadataData | None:
        """
        加载会话元数据（不包括消息列表）

        Args:
            session_id: 会话 ID

        Returns:
            会话元数据字典 或 None（不存在）
        """
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> None:
        """
        删除会话（包括所有消息）

        Args:
            session_id: 会话 ID
        """
        pass

    @abstractmethod
    async def session_exists(self, session_id: str) -> bool:
        """
        检查会话是否存在

        Args:
            session_id: 会话 ID

        Returns:
            是否存在
        """
        pass

    @abstractmethod
    async def list_sessions(self) -> list[str]:
        """
        列出所有会话 ID

        Returns:
            会话 ID 列表
        """
        pass

    @abstractmethod
    async def save_message(self, session_id: str, message: Message) -> None:
        """
        保存单条消息（增量保存）

        Args:
            session_id: 会话 ID
            message: Message 领域对象，序列化策略由实现方决定
        """
        pass

    @abstractmethod
    async def load_messages(
        self, session_id: str, *, from_message_id: str | None = None
    ) -> list[Message]:
        """
        加载会话消息。

        Args:
            session_id: 会话 ID
            from_message_id: 若提供，只返回该消息（含）之后的消息；否则返回全部。

        Returns:
            Message 对象列表
        """
        pass

    @abstractmethod
    async def delete_messages(self, session_id: str) -> None:
        """
        删除会话的所有消息

        Args:
            session_id: 会话 ID
        """
        pass

    @abstractmethod
    async def save_artifact(
        self,
        session_id: str,
        kind: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> ArtifactMetadata:
        """
        保存工具执行产生的大文本产物（artifact）。

        默认实现抛出 NotImplementedError，具体存储可按需覆盖。
        """
        raise NotImplementedError("save_artifact is not implemented")

    @abstractmethod
    async def read_artifact(
        self,
        artifact_ref: str,
        offset: int = 0,
        limit: int = 50000,
    ) -> ArtifactReadResult:
        """
        读取 artifact 内容（支持分段读取）。

        默认实现抛出 NotImplementedError，具体存储可按需覆盖。
        """
        raise NotImplementedError("read_artifact is not implemented")

    @abstractmethod
    async def delete_artifacts(self, session_id: str) -> None:
        """
        删除会话相关的所有 artifact。

        默认实现抛出 NotImplementedError，具体存储可按需覆盖。
        """
        raise NotImplementedError("delete_artifacts is not implemented")
