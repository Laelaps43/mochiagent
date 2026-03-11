"""
Message Info - 消息元数据定义
"""

from typing import Annotated, Any, Dict, Literal, Optional, Union
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
    code: Optional[str] = None
    hint: Optional[str] = None


class MessageInfoBase(BaseModel):
    """消息元数据基类"""

    id: str
    session_id: str
    role: Literal["user", "assistant", "system"]
    created_at: int = 0


class UserMessageInfo(MessageInfoBase):
    """用户消息元数据"""

    role: Literal["user"] = "user"
    agent: str = "general"
    model: Optional[ModelRef] = None
    system: Optional[str] = None
    tools: Optional[Dict[str, bool]] = None
    variant: Optional[str] = None


class AssistantMessageInfo(MessageInfoBase):
    """AI 助手消息元数据"""

    role: Literal["assistant"] = "assistant"
    parent_id: str
    agent: str = "general"
    model_id: str
    provider_id: str
    path: Optional[PathRef] = None
    summary: Optional[bool] = None
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    completed_at: Optional[int] = None
    finish: Optional[str] = None
    error: Optional[ErrorRef] = None


class SystemMessageInfo(MessageInfoBase):
    """系统消息元数据（轻量，不持久化）"""

    id: str = ""
    session_id: str = ""
    role: Literal["system"] = "system"
    created_at: int = 0


MessageInfo = Annotated[
    Union[UserMessageInfo, AssistantMessageInfo, SystemMessageInfo],
    Discriminator("role"),
]
