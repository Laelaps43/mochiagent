"""Tool-related types."""

from typing import Literal

from pydantic import BaseModel, Field


class ToolFunctionPayload(BaseModel):
    name: str = ""
    arguments: str = ""


class ToolCallPayload(BaseModel):
    id: str = ""
    type: Literal["function"] = "function"
    function: ToolFunctionPayload = Field(default_factory=ToolFunctionPayload)


class ToolResult(BaseModel):
    """工具执行结果"""

    tool_call_id: str
    tool_name: str
    result: object
    """Tool execution output. Typically a dict, str, or list returned by Tool.execute()."""
    error: str | None = None
    success: bool = True
    summary: str | None = None
    artifact_ref: str | None = None
    artifact_path: str | None = None
    raw_size_chars: int | None = None
    truncated: bool = False
