"""
Part Models - Part 数据模型定义
"""

import time
from typing import Any, Dict, Literal, Optional, Type, Union
from uuid import uuid4

from pydantic import BaseModel

from agent.constants import UUID_PREFIX_LENGTH
from agent.types import ToolResult


class TimeInfo(BaseModel):
    """时间信息"""

    start: int  # 毫秒时间戳
    end: Optional[int] = None  # 毫秒时间戳


class PartBase(BaseModel):
    """Part 基础字段"""

    id: str
    session_id: str
    message_id: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any], session_id: str, message_id: str) -> "Part":
        """
        从字典创建 Part（子类必须实现）

        Args:
            data: Part 数据字典（不包含 type 字段）
            session_id: 会话 ID
            message_id: 消息 ID

        Returns:
            Part 实例
        """
        raise NotImplementedError(f"{cls.__name__} must implement from_dict()")

    def to_llm_format(self) -> Optional[Dict[str, Any]]:
        """
        转换为 LLM 格式的贡献

        返回 None 表示此 Part 不参与 LLM 消息构建
        返回字典表示此 Part 对 LLM 消息的贡献

        Returns:
            None 或包含 type 和其他字段的字典
        """
        return None  # 默认不参与


# ============ TextPart - 文本内容 ============


class TextPart(PartBase):
    """文本内容 Part"""

    type: Literal["text"] = "text"
    text: str
    synthetic: Optional[bool] = None  # 是否为系统生成
    ignored: Optional[bool] = None  # 是否忽略
    time: Optional[TimeInfo] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def create_fast(
        cls,
        *,
        session_id: str,
        message_id: str,
        text: str,
        synthetic: Optional[bool] = None,
        ignored: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "TextPart":
        """热路径创建：跳过 pydantic 校验，降低流式开销。"""
        return cls.model_construct(
            id=f"part_{uuid4().hex[:UUID_PREFIX_LENGTH]}",
            session_id=session_id,
            message_id=message_id,
            type="text",
            text=text,
            synthetic=synthetic,
            ignored=ignored,
            time=TimeInfo.model_construct(start=int(time.time() * 1000), end=None),
            metadata=metadata,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], session_id: str, message_id: str) -> "TextPart":
        """从字典创建 TextPart"""
        return cls(
            id=f"part_{uuid4().hex[:UUID_PREFIX_LENGTH]}",
            session_id=session_id,
            message_id=message_id,
            text=data["text"],
            synthetic=data.get("synthetic"),
            ignored=data.get("ignored"),
            time=TimeInfo(start=int(time.time() * 1000)),
            metadata=data.get("metadata"),
        )

    def to_llm_format(self) -> Optional[Dict[str, Any]]:
        """文本内容贡献给 LLM 消息的 content"""
        return {"type": "text", "content": self.text}

    def to_event_payload(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "message_id": self.message_id,
            "type": "text",
            "text": self.text,
            "synthetic": self.synthetic,
            "ignored": self.ignored,
            "time": {"start": self.time.start, "end": self.time.end} if self.time else None,
            "metadata": self.metadata,
        }


# ============ ReasoningPart - AI 思考过程 ============


class ReasoningPart(PartBase):
    """AI 思考过程 Part"""

    type: Literal["reasoning"] = "reasoning"
    text: str
    time: TimeInfo
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def create_fast(
        cls,
        *,
        session_id: str,
        message_id: str,
        text: str,
        start: int,
        end: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ReasoningPart":
        """热路径创建：跳过 pydantic 校验，降低流式开销。"""
        return cls.model_construct(
            id=f"part_{uuid4().hex[:UUID_PREFIX_LENGTH]}",
            session_id=session_id,
            message_id=message_id,
            type="reasoning",
            text=text,
            time=TimeInfo.model_construct(start=start, end=end),
            metadata=metadata,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], session_id: str, message_id: str) -> "ReasoningPart":
        """从字典创建 ReasoningPart"""
        return cls(
            id=f"part_{uuid4().hex[:UUID_PREFIX_LENGTH]}",
            session_id=session_id,
            message_id=message_id,
            text=data["text"],
            time=TimeInfo(**data["time"])
            if "time" in data
            else TimeInfo(start=int(time.time() * 1000)),
            metadata=data.get("metadata"),
        )

    def to_llm_format(self) -> Optional[Dict[str, Any]]:
        """思考过程不发送给 LLM（内部数据）"""
        return None

    def to_event_payload(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "message_id": self.message_id,
            "type": "reasoning",
            "text": self.text,
            "time": {"start": self.time.start, "end": self.time.end},
            "metadata": self.metadata,
        }


# ============ ToolPart - 工具调用 ============


class ToolStatePending(BaseModel):
    """工具状态 - pending"""

    status: Literal["pending"] = "pending"
    input: Dict[str, Any]
    raw: str


class ToolStateRunning(BaseModel):
    """工具状态 - running"""

    status: Literal["running"] = "running"
    input: Dict[str, Any]
    title: Optional[str] = None
    metadata: Optional[Any] = None
    time: Dict[str, int]  # {"start": timestamp}


class ToolStateCompleted(BaseModel):
    """工具状态 - completed"""

    status: Literal["completed"] = "completed"
    input: Dict[str, Any]
    output: str
    summary: str = ""
    artifact_ref: Optional[str] = None
    artifact_path: Optional[str] = None
    raw_size_chars: Optional[int] = None
    truncated: bool = False
    title: str
    metadata: Any
    time: Dict[str, int]  # {"start": timestamp, "end": timestamp}


class ToolStateError(BaseModel):
    """工具状态 - error"""

    status: Literal["error"] = "error"
    input: Dict[str, Any]
    error: str
    metadata: Optional[Any] = None
    time: Dict[str, int]  # {"start": timestamp, "end": timestamp}


ToolState = Union[ToolStatePending, ToolStateRunning, ToolStateCompleted, ToolStateError]


class ToolPart(PartBase):
    """工具调用 Part"""

    type: Literal["tool"] = "tool"
    call_id: str
    tool: str
    state: ToolState
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def create_running(cls, session_id: str, message_id: str, tool_call: dict) -> "ToolPart":
        """
        创建 running 状态的 ToolPart

        Args:
            session_id: 会话ID
            message_id: 消息ID
            tool_call: 工具调用字典

        Returns:
            ToolPart
        """
        call_id = tool_call["id"]
        function_info = tool_call.get("function", {})
        tool_name = function_info.get("name", "unknown")
        arguments = function_info.get("arguments", "{}")

        return cls.model_construct(
            id=f"part_{uuid4().hex[:UUID_PREFIX_LENGTH]}",
            session_id=session_id,
            message_id=message_id,
            type="tool",
            call_id=call_id,
            tool=tool_name,
            state=ToolStateRunning.model_construct(
                status="running",
                input={"arguments": arguments},
                title=tool_name,
                metadata=None,
                time={"start": int(time.time() * 1000)},
            ),
            metadata=None,
        )

    def update_to_completed(self, result: ToolResult) -> "ToolPart":
        """
        更新为 completed 状态

        Args:
            result: 工具执行结果

        Returns:
            更新后的 ToolPart
        """
        import json

        # 优先使用后处理得到的摘要；未提供时回退到 result 序列化文本。
        if result.summary is not None:
            output = result.summary
        elif isinstance(result.result, str):
            output = result.result
        else:
            try:
                output = json.dumps(result.result, ensure_ascii=False)
            except (TypeError, ValueError):
                output = str(result.result)

        return ToolPart.model_construct(
            id=self.id,
            session_id=self.session_id,
            message_id=self.message_id,
            type="tool",
            call_id=result.tool_call_id,
            tool=self.tool,
            state=ToolStateCompleted.model_construct(
                status="completed",
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
                    "error": result.error,
                    "success": result.success,
                    "artifact_ref": result.artifact_ref,
                    "artifact_path": result.artifact_path,
                    "raw_size_chars": result.raw_size_chars,
                    "truncated": result.truncated,
                },
                time={
                    "start": self.state.time["start"],
                    "end": int(time.time() * 1000),
                },
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
        return ToolPart.model_construct(
            id=self.id,
            session_id=self.session_id,
            message_id=self.message_id,
            type="tool",
            call_id=result.tool_call_id,
            tool=self.tool,
            state=ToolStateError.model_construct(
                status="error",
                input=self.state.input,
                error=result.error or "Unknown error",
                metadata={
                    "tool_call_id": result.tool_call_id,
                    "tool_name": result.tool_name,
                    "result": result.result,
                    "error": result.error,
                    "success": result.success,
                },
                time={
                    "start": self.state.time["start"],
                    "end": int(time.time() * 1000),
                },
            ),
            metadata=self.metadata,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], session_id: str, message_id: str) -> "ToolPart":
        """
        从字典创建 ToolPart

        注意：ToolPart 通常由系统创建，不从用户输入创建
        """
        raise NotImplementedError(
            "ToolPart should be created programmatically, not from user input"
        )

    def to_llm_format(self) -> Optional[Dict[str, Any]]:
        """
        工具调用贡献给 LLM 消息

        根据状态返回不同的内容：
        - running/completed/error: 返回 tool_call
        - completed: 额外返回 tool_result
        - error: 额外返回 tool_result (错误信息)
        """
        result: Dict[str, Any] = {"type": "tool"}

        # 1. tool_call（running、completed 或 error 状态）
        if self.state.status in ["running", "completed", "error"]:
            result["tool_call"] = {
                "id": self.call_id,
                "type": "function",
                "function": {
                    "name": self.tool,
                    "arguments": self.state.input.get("arguments", "{}"),
                },
            }

        # 2. tool_result（completed 或 error 状态）
        if self.state.status == "completed":
            result["tool_result"] = {
                "role": "tool",
                "content": self.state.summary or self.state.output,
                "tool_call_id": self.call_id,
            }
        elif self.state.status == "error":
            # 使用 getattr 安全访问，防止类型检查问题
            error_msg = getattr(self.state, "error", "Unknown error")
            result["tool_result"] = {
                "role": "tool",
                "content": f"Error: {error_msg}",
                "tool_call_id": self.call_id,
            }

        return result if len(result) > 1 else None  # 只有 type 时返回 None

    def to_event_payload(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "message_id": self.message_id,
            "type": "tool",
            "call_id": self.call_id,
            "tool": self.tool,
            "state": self.state.model_dump(),
            "metadata": self.metadata,
        }


# ============ Part Union Type ============

Part = Union[TextPart, ReasoningPart, ToolPart]


# ============ Part Registry & Factory ============

# Part 类型注册表（用于从字典创建 Part）
PART_REGISTRY: Dict[str, Type[PartBase]] = {
    "text": TextPart,
    "reasoning": ReasoningPart,
    # "tool" 不包含在注册表中，因为它不从用户输入创建
    # 将来加图片时只需添加一行：
    # "image": ImagePart,
}


def create_part_from_dict(
    data: Dict[str, Any],
    session_id: str,
    message_id: str,
) -> Part:
    """
    Part 工厂函数 - 从字典创建 Part（多态）

    Args:
        data: Part 数据字典，必须包含 "type" 字段
        session_id: 会话 ID
        message_id: 消息 ID

    Returns:
        Part 实例

    Raises:
        ValueError: 未知的 part type

    Examples:
        >>> part = create_part_from_dict(
        ...     {"type": "text", "text": "hello"},
        ...     session_id="xxx",
        ...     message_id="yyy"
        ... )
    """
    part_type = data.get("type")

    # 从注册表获取 Part 类
    part_class = PART_REGISTRY.get(part_type)
    if not part_class:
        raise ValueError(f"Unknown part type: {part_type}")

    # 调用对应 Part 类的 from_dict 方法（多态）
    return part_class.from_dict(data, session_id, message_id)
