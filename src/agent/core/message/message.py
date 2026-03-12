"""
Message Container - Message = Info + Parts
"""

from pydantic import BaseModel, Field

from .info import MessageInfo, SystemMessageInfo
from .part import Part, TextPart


class Message(BaseModel):
    """
    消息容器：Message = Info + Parts[]

    一个消息由元数据（Info）和内容片段（Parts）组成
    """

    info: MessageInfo
    parts: list[Part] = Field(default_factory=list)

    @classmethod
    def create_system(cls, content: str) -> "Message":
        """创建系统消息（不持久化，仅用于 LLM 调用）"""
        return cls(
            info=SystemMessageInfo(),
            parts=[TextPart.create_fast(session_id="", message_id="", text=content)],
        )

    def add_part(self, part: Part) -> None:
        """添加 Part 到消息"""
        self.parts.append(part)

    @property
    def message_id(self) -> str:
        return self.info.id

    @property
    def session_id(self) -> str:
        return self.info.session_id

    @property
    def role(self) -> str:
        return self.info.role
