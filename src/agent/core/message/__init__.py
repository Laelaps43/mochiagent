"""
Message Module - 消息和 Part 模型
"""

from .part import (
    Part,
    PartBase,
    UserInput,
    UserTextInput,
    UserReasoningInput,
    TextPart,
    ReasoningPart,
    ToolPart,
    ToolState,
    ToolStatePending,
    ToolStateRunning,
    ToolStateCompleted,
    ToolStateError,
    TimeInfo,
    create_part_from_user_input,
)
from .info import MessageInfo, UserMessageInfo, AssistantMessageInfo
from .message import Message, SerializedMessageData

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
    "UserReasoningInput",
    # Tool State
    "ToolState",
    "ToolStatePending",
    "ToolStateRunning",
    "ToolStateCompleted",
    "ToolStateError",
    "TimeInfo",
    # Part factory
    "create_part_from_user_input",
    # Message types
    "MessageInfo",
    "UserMessageInfo",
    "AssistantMessageInfo",
    "Message",
    "SerializedMessageData",
]
