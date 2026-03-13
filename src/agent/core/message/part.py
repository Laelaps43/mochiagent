"""
Part Models - Part 数据模型定义
"""

import json
from typing import Annotated, Literal

from pydantic import BaseModel, BaseModel as _PydanticBaseModel, Discriminator, Field

from agent.core.utils import gen_id, now_ms
from agent.types import ToolCallPayload, ToolResult


class TimeInfo(BaseModel):
    """时间信息"""

    start: int  # 毫秒时间戳
    end: int | None = None  # 毫秒时间戳


class PartBase(BaseModel):
    """Part 基础字段"""

    id: str = Field(default_factory=lambda: gen_id("part_"))
    session_id: str
    message_id: str


# ============ User Input (简单 DTO，无 id) ============


class UserTextInput(BaseModel):
    """用户输入文本（无 id/session_id，进入 Session 后转为 TextPart）"""

    type: Literal["text"] = "text"
    text: str
    synthetic: bool | None = None
    ignored: bool | None = None
    metadata: dict[str, object] | None = None

    def to_part(self, session_id: str, message_id: str) -> "TextPart":
        return TextPart(
            session_id=session_id,
            message_id=message_id,
            text=self.text,
            synthetic=self.synthetic,
            ignored=self.ignored,
            metadata=self.metadata,
        )


UserInput = UserTextInput


# ============ TextPart - 文本内容 ============


class TextPart(PartBase):
    """文本内容 Part"""

    type: Literal["text"] = "text"
    text: str
    synthetic: bool | None = None  # 是否为系统生成
    ignored: bool | None = None  # 是否忽略
    time: TimeInfo = Field(default_factory=lambda: TimeInfo(start=now_ms()))
    metadata: dict[str, object] | None = None


# ============ ReasoningPart - AI 思考过程 ============


class ReasoningPart(PartBase):
    """AI 思考过程 Part"""

    type: Literal["reasoning"] = "reasoning"
    text: str
    time: TimeInfo
    metadata: dict[str, object] | None = None


# ============ ToolPart - 工具调用 ============


class ToolInput(BaseModel):
    """工具调用输入"""

    arguments: str = "{}"


class ToolStatePending(BaseModel):
    """工具状态 - pending"""

    status: Literal["pending"] = "pending"
    input: ToolInput
    raw: str


class ToolStateRunning(BaseModel):
    """工具状态 - running"""

    status: Literal["running"] = "running"
    input: ToolInput
    title: str | None = None
    metadata: dict[str, object] | None = None
    time: TimeInfo


class ToolStateCompleted(BaseModel):
    """工具状态 - completed"""

    status: Literal["completed"] = "completed"
    input: ToolInput
    output: str
    summary: str = ""
    artifact_ref: str | None = None
    artifact_path: str | None = None
    raw_size_chars: int | None = None
    truncated: bool = False
    title: str
    metadata: dict[str, object] | None = None
    time: TimeInfo


class ToolStateError(BaseModel):
    """工具状态 - error"""

    status: Literal["error"] = "error"
    input: ToolInput
    error: str
    metadata: dict[str, object] | None = None
    time: TimeInfo


ToolState = Annotated[
    ToolStatePending | ToolStateRunning | ToolStateCompleted | ToolStateError,
    Discriminator("status"),
]


class ToolPart(PartBase):
    """工具调用 Part"""

    type: Literal["tool"] = "tool"
    call_id: str
    tool: str
    state: ToolState
    metadata: dict[str, object] | None = None

    @classmethod
    def create_running(
        cls, session_id: str, message_id: str, tool_call: "ToolCallPayload"
    ) -> "ToolPart":
        name = tool_call.function.name or "unknown"
        return cls(
            session_id=session_id,
            message_id=message_id,
            call_id=tool_call.id,
            tool=name,
            state=ToolStateRunning(
                input=ToolInput(arguments=tool_call.function.arguments or "{}"),
                title=name,
                time=TimeInfo(start=now_ms()),
            ),
        )

    def update_to_completed(self, result: ToolResult) -> "ToolPart":
        """
        更新为 completed 状态

        Args:
            result: 工具执行结果

        Returns:
            更新后的 ToolPart
        """
        # 优先使用后处理得到的摘要；未提供时回退到 result 序列化文本。
        if result.summary is not None:
            output = result.summary
        elif isinstance(result.result, str):
            output = result.result
        elif isinstance(result.result, _PydanticBaseModel):
            output = json.dumps(result.result.model_dump(), ensure_ascii=False)
        else:
            try:
                output = json.dumps(result.result, ensure_ascii=False)
            except (TypeError, ValueError):
                output = str(result.result)

        if not isinstance(self.state, ToolStateRunning):
            raise ValueError(
                f"Cannot complete a ToolPart in '{self.state.status}' state; expected 'running'"
            )

        return ToolPart(
            id=self.id,
            session_id=self.session_id,
            message_id=self.message_id,
            call_id=result.tool_call_id,
            tool=self.tool,
            state=ToolStateCompleted(
                input=self.state.input,
                output=output,
                summary=output,
                artifact_ref=result.artifact_ref,
                artifact_path=result.artifact_path,
                raw_size_chars=result.raw_size_chars,
                truncated=result.truncated,
                title=self.tool,
                metadata={
                    "tool_call_id": result.tool_call_id,
                    "tool_name": result.tool_name,
                },
                time=TimeInfo(
                    start=self.state.time.start,
                    end=now_ms(),
                ),
            ),
            metadata=self.metadata,
        )

    def update_to_error(self, result: ToolResult) -> "ToolPart":
        """
        更新为 error 状态

        Args:
            result: 工具执行结果（包含错误信息）

        Returns:
            更新后的 ToolPart
        """
        if not isinstance(self.state, ToolStateRunning):
            raise ValueError(
                f"Cannot error a ToolPart in '{self.state.status}' state; expected 'running'"
            )

        return ToolPart(
            id=self.id,
            session_id=self.session_id,
            message_id=self.message_id,
            call_id=result.tool_call_id,
            tool=self.tool,
            state=ToolStateError(
                input=self.state.input,
                error=result.error or "Unknown error",
                metadata={
                    "tool_call_id": result.tool_call_id,
                    "tool_name": result.tool_name,
                    "result": str(result.result) if result.result is not None else None,
                    "error": result.error,
                    "success": result.success,
                },
                time=TimeInfo(
                    start=self.state.time.start,
                    end=now_ms(),
                ),
            ),
            metadata=self.metadata,
        )


# ============ Part Union Type ============

# 完整 Part（存储/流转层，带 id/session_id/message_id）
Part = Annotated[
    TextPart | ReasoningPart | ToolPart,
    Discriminator("type"),
]
