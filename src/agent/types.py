"""
Agent Framework Core Types
定义框架中使用的所有核心类型
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import override

from pydantic import BaseModel, Field, SecretStr, field_validator


class MessageRole(str, Enum):
    """消息角色"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    COMPACTION = "compaction"


class SessionState(str, Enum):
    """会话状态"""

    IDLE = "idle"
    PROCESSING = "processing"
    STREAMING = "streaming"
    WAITING_TOOL = "waiting_tool"
    ERROR = "error"
    TERMINATED = "terminated"


class EventType(str, Enum):
    """事件类型"""

    # 会话事件
    SESSION_CREATED = "session.created"
    SESSION_STATE_CHANGED = "session.state_changed"
    SESSION_AGENT_SWITCHED = "session.agent_switched"
    SESSION_TERMINATED = "session.terminated"

    # 消息事件
    MESSAGE_RECEIVED = "message.received"
    PART_CREATED = "part.created"
    MESSAGE_DONE = "message.done"

    # 压缩事件
    CONTEXT_COMPACTING = "context.compacting"
    CONTEXT_COMPACTED = "context.compacted"

    # LLM 事件
    LLM_ERROR = "llm.error"
    LLM_THINKING = "llm.thinking"
    LLM_RETRY = "llm.retry"

    # SubAgent 事件
    SUBAGENT_INVOKED = "subagent.invoked"
    SUBAGENT_COMPLETED = "subagent.completed"

    # 取消事件（非破坏性，不删数据）
    SESSION_CANCELLED = "session.cancelled"


class Event(BaseModel):
    """运行时事件对象"""

    type: EventType
    session_id: str
    data: dict[str, object] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    metadata: dict[str, object] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


class LLMConfig(BaseModel):
    """LLM配置"""

    adapter: str
    provider: str
    model: str
    api_key: SecretStr | None = None
    base_url: str | None = None

    @override
    def __repr__(self) -> str:
        return (
            f"LLMConfig(adapter={self.adapter!r}, provider={self.provider!r}, "
            f"model={self.model!r}, api_key='***', base_url={self.base_url!r})"
        )

    @override
    def __str__(self) -> str:
        return self.__repr__()

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"base_url must start with http:// or https://, got: {v!r}")
        return v

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = None
    context_window_tokens: int | None = None
    stream: bool = True
    timeout: int = 60
    openai_max_retries: int = 2
    max_overflow_retries: int = 3
    extra_params: dict[str, object] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    """工具定义"""

    name: str
    description: str
    parameters: dict[str, object]  # JSON Schema
    required: list[str] = Field(default_factory=list)


class StreamChunk(BaseModel):
    """流式输出片段"""

    session_id: str
    content: str
    finish_reason: str | None = None
    tool_calls: list[object] | None = None
    is_final: bool = False
