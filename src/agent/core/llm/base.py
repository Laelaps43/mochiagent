"""
LLM Provider Base - LLM提供商基类
定义统一的LLM接口
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, List, Optional

from agent.core.message import Message as InternalMessage
from agent.types import LLMConfig, LLMStreamChunk, ToolDefinition


class LLMProvider(ABC):
    """
    LLM提供商基类
    所有LLM适配器必须继承此类
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    def prepare_messages(self, messages: list[InternalMessage]) -> list[dict[str, Any]]:
        """将内部 Message 列表转换为 LLM API 格式的 dict 列表。

        子类可覆写（如 Claude API 需要 system 单独传）。
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            text_contents: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []

            for part in msg.parts:
                match part:
                    case {"type": "text", "text": str(text)}:
                        text_contents.append(text)
                    case {
                        "type": "tool",
                        "call_id": str(call_id),
                        "tool": str(tool),
                        "state": state,
                    }:
                        if state.status in ("running", "completed", "error"):
                            tool_calls.append(
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": tool, "arguments": state.input.arguments},
                                }
                            )
                        if state.status == "completed":
                            tool_results.append(
                                {
                                    "role": "tool",
                                    "content": state.summary or state.output,
                                    "tool_call_id": call_id,
                                }
                            )
                        elif state.status == "error":
                            tool_results.append(
                                {
                                    "role": "tool",
                                    "content": f"Error: {state.error or 'Unknown error'}",
                                    "tool_call_id": call_id,
                                }
                            )

            if not text_contents and not tool_calls:
                continue

            main_msg: dict[str, Any] = {
                "role": msg.info.role,
                "content": "".join(text_contents),
            }
            if msg.info.role == "assistant" and tool_calls:
                main_msg["tool_calls"] = tool_calls
            result.append(main_msg)
            result.extend(tool_results)

        return result

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[InternalMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamChunk]:
        """
        流式对话

        Args:
            messages: 内部 Message 列表（含 system/user/assistant）
            tools: 可用工具列表
            **kwargs: 额外参数

        Yields:
            Dict包含:
            - content: str  # 文本内容
            - tool_calls: List[Dict]  # 工具调用
            - finish_reason: str  # 结束原因
        """
        pass

    @abstractmethod
    async def complete(
        self,
        messages: list[InternalMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> LLMStreamChunk:
        """
        非流式对话

        Args:
            messages: 内部 Message 列表（含 system/user/assistant）
            tools: 可用工具列表
            **kwargs: 额外参数

        Returns:
            Dict包含:
            - content: str
            - tool_calls: List[Dict]
            - finish_reason: str
        """
        pass
