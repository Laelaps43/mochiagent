"""
Message Info - 消息元数据定义
"""

from typing import Annotated, Literal
from pydantic import BaseModel, Discriminator, Field

from agent.types import TokenUsage


class ModelRef(BaseModel):
    """模型引用"""

    provider_id: str = ""
    model_id: str = ""


class PathRef(BaseModel):
    """路径引用"""

    cwd: str = ""
    root: str = ""


class ErrorRef(BaseModel):
    """错误信息"""

    message: str = ""
    code: str | None = None
    hint: str | None = None


class MessageInfoBase(BaseModel):
    """消息元数据基类"""

    id: str
    session_id: str
    created_at: int = 0


class UserMessageInfo(MessageInfoBase):
    """用户消息元数据"""

    role: Literal["user"] = "user"
    agent: str = "general"
    model: ModelRef | None = None
    system: str | None = None
    tools: dict[str, bool] | None = None
    variant: str | None = None


class AssistantMessageInfo(MessageInfoBase):
    """AI 助手消息元数据"""

    role: Literal["assistant"] = "assistant"
    parent_id: str
    agent: str = "general"
    model_id: str
    provider_id: str
    path: PathRef | None = None
    summary: bool | None = None
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    completed_at: int | None = None
    finish: str | None = None
    error: ErrorRef | None = None


class SystemMessageInfo(MessageInfoBase):
    """系统消息元数据（轻量，不持久化）"""

    id: str = ""
    session_id: str = ""
    role: Literal["system"] = "system"
    created_at: int = 0


class CompactionMessageInfo(MessageInfoBase):
    """压缩书签消息元数据"""

    role: Literal["compaction"] = "compaction"
    compacted_count: int = 0
    compaction_metadata: dict[str, object] = Field(default_factory=dict)


MessageInfo = Annotated[
    UserMessageInfo | AssistantMessageInfo | SystemMessageInfo | CompactionMessageInfo,
    Discriminator("role"),
]
