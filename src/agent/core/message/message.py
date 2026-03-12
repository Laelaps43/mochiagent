"""
Message Container - Message = Info + Parts
"""

from pydantic import BaseModel, Field

from agent.core.utils import gen_id, now_ms

from .info import CompactionMessageInfo, MessageInfo, SystemMessageInfo
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
        parts: list[Part] = [TextPart(session_id="", message_id="", text=content)]
        return cls(info=SystemMessageInfo(), parts=parts)

    @classmethod
    def create_compaction(
        cls,
        *,
        session_id: str,
        summary: str,
        compacted_count: int = 0,
        compaction_metadata: dict[str, object] | None = None,
    ) -> "Message":
        """创建压缩书签消息（摘要存在 TextPart 中）"""
        message_id = gen_id("cmp_")
        return cls(
            info=CompactionMessageInfo(
                id=message_id,
                session_id=session_id,
                created_at=now_ms(),
                compacted_count=compacted_count,
                compaction_metadata=compaction_metadata or {},
            ),
            parts=[
                TextPart(
                    session_id=session_id,
                    message_id=message_id,
                    text=f"COMPACTION_SUMMARY\n{summary}",
                )
            ],
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
