"""
Agent Framework Core Types
定义框架中使用的所有核心类型
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import time
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agent.core.message.message import SerializedMessageData


class MessageRole(str, Enum):
    """消息角色"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolFunctionPayload(TypedDict):
    name: str
    arguments: str


class ToolCallPayload(TypedDict):
    id: str
    type: Literal["function"]
    function: ToolFunctionPayload


class Message(BaseModel):
    """标准消息格式"""

    role: MessageRole
    content: str
    tool_calls: Optional[List[ToolCallPayload]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # For tool responses


class ToolResult(BaseModel):
    """工具执行结果"""

    tool_call_id: str
    tool_name: str
    result: Any
    error: Optional[str] = None
    success: bool = True
    summary: Optional[str] = None
    artifact_ref: Optional[str] = None
    artifact_path: Optional[str] = None
    raw_size_chars: Optional[int] = None
    truncated: bool = False


ContextBudgetSource = Literal["estimated", "provider"]


@dataclass
class ContextBudgetData:
    total_tokens: int | None
    used_tokens: int
    remaining_tokens: int | None
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    source: ContextBudgetSource
    updated_at_ms: int


@dataclass
class SessionMetadataData:
    session_id: str
    state: str
    model_profile_id: str
    agent_name: str
    context_budget: ContextBudgetData
    created_at: str
    updated_at: str


@dataclass
class SessionData:
    session_id: str
    state: str
    model_profile_id: str
    agent_name: str
    context_budget: ContextBudgetData
    message_count: int
    messages: list["SerializedMessageData"]
    created_at: str
    updated_at: str


@dataclass(slots=True)
class ContextBudget:
    """上下文窗口预算快照"""

    total_tokens: int | None = None
    used_tokens: int = 0
    remaining_tokens: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    source: ContextBudgetSource = "estimated"
    updated_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> ContextBudgetData:
        return ContextBudgetData(
            total_tokens=self.total_tokens,
            used_tokens=self.used_tokens,
            remaining_tokens=self.remaining_tokens,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            reasoning_tokens=self.reasoning_tokens,
            source=self.source,
            updated_at_ms=self.updated_at_ms,
        )


class TokenUsage(TypedDict):
    input: int
    output: int
    reasoning: int



class LLMStreamChunk(TypedDict, total=False):
    content: str
    thinking: str
    tool_calls: List[ToolCallPayload]
    finish_reason: str
    usage: Dict[str, Any]


class LLMMessage(TypedDict, total=False):
    role: str
    content: str
    tool_calls: List[ToolCallPayload]
    tool_call_id: str
    name: str


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
    SESSION_CREATED = "session.created"  # 会话已创建（内部通知）
    SESSION_STATE_CHANGED = "session.state_changed"
    SESSION_AGENT_SWITCHED = "session.agent_switched"  # Agent 切换
    SESSION_TERMINATED = "session.terminated"

    # 消息事件
    MESSAGE_RECEIVED = "message.received"  # 接收用户消息
    PART_CREATED = "part.created"  # Part 创建（流式发送 Part）
    MESSAGE_DONE = "message.done"  # 消息完成

    # 错误事件
    LLM_ERROR = "llm.error"
    LLM_THINKING = "llm.thinking"


@dataclass(slots=True)
class Event:
    """运行时事件对象（热路径使用轻量 dataclass）"""

    type: EventType
    session_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMConfig(BaseModel):
    """LLM配置"""

    adapter: str  # openai_compatible, anthropic, etc.
    provider: str  # openai, zhipu, kimi, etc.
    model: str  # gpt-4, claude-3-sonnet, etc.
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    context_window_tokens: Optional[int] = None
    stream: bool = True
    timeout: int = 60
    openai_max_retries: Optional[int] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    """工具定义"""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    required: List[str] = Field(default_factory=list)


class StreamChunk(BaseModel):
    """流式输出片段"""

    session_id: str
    content: str
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[ToolCallPayload]] = None
    is_final: bool = False
