"""
Message Module - 消息和 Part 模型
"""

from .part import (
    Part,
    PartBase,
    UserInput,
    UserTextInput,
    TextPart,
    ReasoningPart,
    ToolPart,
    ToolInput,
    ToolState,

    ToolStatePending,
    ToolStateRunning,
    ToolStateCompleted,
    ToolStateError,
    TimeInfo,
)
from .info import (
    MessageInfo,
    ModelRef,
    PathRef,
    ErrorRef,
    UserMessageInfo,
    AssistantMessageInfo,
    SystemMessageInfo,
)
from .message import Message

__all__ = [
    # Part types - 完整实体（存储/流转层）
    "Part",
    "PartBase",
    "TextPart",
    "ReasoningPart",
    "ToolPart",
    # Part types - 用户输入（简单 DTO）
    "UserInput",
    "UserTextInput",
    # Tool State
    "ToolInput",
    "ToolState",
    "ToolStatePending",
    "ToolStateRunning",
    "ToolStateCompleted",
    "ToolStateError",
    "TimeInfo",
    # Part factory
    # Message types
    "MessageInfo",
    "ModelRef",
    "PathRef",
    "ErrorRef",
    "UserMessageInfo",
    "AssistantMessageInfo",
    "SystemMessageInfo",
    "Message",
]
