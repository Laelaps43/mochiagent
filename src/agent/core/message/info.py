"""
Message Info - 消息元数据定义
"""

from typing import Any, Dict, Literal, Optional, Union
from pydantic import BaseModel, Field


class MessageInfoBase(BaseModel):
    """消息元数据基类"""

    id: str
    session_id: str
    role: Literal["user", "assistant"]
    time: Dict[str, int] = Field(default_factory=dict)  # {"created": ts, "completed": ts}


class UserMessageInfo(MessageInfoBase):
    """用户消息元数据"""

    role: Literal["user"] = "user"
    agent: str = "general"
    model: Optional[Dict[str, str]] = None  # {"provider_id": "openai", "model_id": "gpt-4"}
    system: Optional[str] = None  # 系统提示
    tools: Optional[Dict[str, bool]] = None  # 可用工具
    variant: Optional[str] = None  # 模型变体


class AssistantMessageInfo(MessageInfoBase):
    """AI 助手消息元数据"""

    role: Literal["assistant"] = "assistant"
    parent_id: str  # 父消息 ID（用户消息 ID）
    agent: str = "general"
    model_id: str
    provider_id: str
    path: Optional[Dict[str, str]] = None  # {"cwd": "/path", "root": "/path"}
    summary: Optional[bool] = None  # 是否为摘要消息
    cost: float = 0.0  # 成本（美元）
    tokens: Dict[str, Any] = Field(default_factory=dict)  # Token 统计
    finish: Optional[str] = None  # 结束原因（stop/max_tokens/tool_calls 等）
    error: Optional[Dict[str, Any]] = None  # 错误信息


MessageInfo = Union[UserMessageInfo, AssistantMessageInfo]
