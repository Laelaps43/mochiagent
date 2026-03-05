"""
Storage Provider - 存储抽象接口
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, TypedDict

from agent.types import SerializedMessageData, SessionMetadataData


class ArtifactMetadata(TypedDict, total=False):
    artifact_ref: str
    artifact_id: str
    session_id: str
    kind: str
    size: int
    path: str
    created_at_ms: int
    metadata: dict[str, Any]


class ArtifactReadResult(TypedDict, total=False):
    success: bool
    error: str
    artifact_ref: str
    path: str
    content: str
    offset: int
    limit: int
    next_offset: int
    eof: bool
    size: int


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
            session_data: 会话元数据字典，包含：
                - session_id
                - state
                - model_profile_id
                - metadata
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
    async def save_message(self, session_id: str, message_data: SerializedMessageData) -> None:
        """
        保存单条消息（增量保存）

        Args:
            session_id: 会话 ID
            message_data: 消息数据字典（来自 Message.to_dict()）
        """
        pass

    @abstractmethod
    async def load_messages(self, session_id: str) -> list[SerializedMessageData]:
        """
        加载会话的所有消息

        Args:
            session_id: 会话 ID

        Returns:
            消息数据字典列表
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

    async def save_artifact(
        self,
        session_id: str,
        kind: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ArtifactMetadata:
        """
        保存工具执行产生的大文本产物（artifact）。

        默认实现抛出 NotImplementedError，具体存储可按需覆盖。
        """
        raise NotImplementedError("save_artifact is not implemented")

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

    async def delete_artifacts(self, session_id: str) -> None:
        """
        删除会话相关的所有 artifact。

        默认实现抛出 NotImplementedError，具体存储可按需覆盖。
        """
        raise NotImplementedError("delete_artifacts is not implemented")
