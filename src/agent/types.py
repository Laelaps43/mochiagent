"""
Agent Framework Core Types
定义框架中使用的所有核心类型
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import time
from typing import Literal

from pydantic import BaseModel, Field, SecretStr

from agent.core.utils import to_non_negative_int


class MessageRole(str, Enum):
    """消息角色"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    COMPACTION = "compaction"


class ToolFunctionPayload(BaseModel):
    name: str = ""
    arguments: str = ""


class ToolCallPayload(BaseModel):
    id: str = ""
    type: Literal["function"] = "function"
    function: ToolFunctionPayload = Field(default_factory=ToolFunctionPayload)


class Message(BaseModel):
    """标准消息格式"""

    role: MessageRole
    content: str
    tool_calls: list[ToolCallPayload] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # For tool responses


class ToolResult(BaseModel):
    """工具执行结果"""

    tool_call_id: str
    tool_name: str
    result: object
    error: str | None = None
    success: bool = True
    summary: str | None = None
    artifact_ref: str | None = None
    artifact_path: str | None = None
    raw_size_chars: int | None = None
    truncated: bool = False


ContextBudgetSource = Literal["estimated", "provider"]


class ContextBudget(BaseModel):
    """上下文窗口预算快照"""

    total_tokens: int | None = None
    used_tokens: int = 0
    remaining_tokens: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    source: ContextBudgetSource = "estimated"
    updated_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    def update(
        self,
        *,
        total_tokens: int | None,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        source: ContextBudgetSource,
        updated_at_ms: int | None = None,
    ) -> None:
        self.total_tokens = to_non_negative_int(total_tokens) if total_tokens is not None else None
        self.input_tokens = to_non_negative_int(input_tokens)
        self.output_tokens = to_non_negative_int(output_tokens)
        self.reasoning_tokens = to_non_negative_int(reasoning_tokens)
        self.used_tokens = self.input_tokens + self.output_tokens + self.reasoning_tokens
        if self.total_tokens is None:
            self.remaining_tokens = None
        else:
            self.remaining_tokens = max(self.total_tokens - self.used_tokens, 0)
        self.source = "provider" if source == "provider" else "estimated"
        self.updated_at_ms = updated_at_ms if updated_at_ms is not None else int(time.time() * 1000)


class SessionMetadataData(BaseModel):
    session_id: str
    state: str
    model_profile_id: str
    agent_name: str
    context_budget: ContextBudget
    last_compaction_message_id: str | None = None
    created_at: str
    updated_at: str


class SessionData(BaseModel):
    session_id: str
    state: str
    model_profile_id: str
    agent_name: str
    context_budget: ContextBudget
    message_count: int
    messages: list[object]
    created_at: str
    updated_at: str


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0
    reasoning: int = 0


class ProviderUsage(BaseModel):
    """LLM 供应商返回的 token 用量（统一命名，由 adapter 负责映射）"""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


class LLMStreamChunk(BaseModel):
    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCallPayload] = Field(default_factory=list)
    finish_reason: str = ""
    usage: ProviderUsage | None = None


class LLMMessage(BaseModel):
    role: str = ""
    content: str = ""
    tool_calls: list[ToolCallPayload] = Field(default_factory=list)
    tool_call_id: str = ""
    name: str = ""


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

    # 错误事件
    LLM_ERROR = "llm.error"
    LLM_THINKING = "llm.thinking"


class Event(BaseModel):
    """运行时事件对象"""

    type: EventType
    session_id: str
    data: dict[str, object] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    metadata: dict[str, object] = Field(default_factory=dict)


class LLMConfig(BaseModel):
    """LLM配置"""

    adapter: str
    provider: str
    model: str
    api_key: SecretStr | None = None
    base_url: str | None = None
    temperature: float = 0.7
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
    tool_calls: list[ToolCallPayload] | None = None
    is_final: bool = False
