"""
LLM Provider Base - LLM提供商基类
定义统一的LLM接口
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from agent.core.message import Message as InternalMessage
from agent.core.message.part import TextPart, ToolPart
from agent.types import LLMConfig, LLMStreamChunk, ToolDefinition


class LLMProvider(ABC):
    """
    LLM提供商基类
    所有LLM适配器必须继承此类
    """

    def __init__(self, config: LLMConfig):
        self.config: LLMConfig = config

    def prepare_messages(self, messages: list[InternalMessage]) -> list[dict[str, object]]:
        """将内部 Message 列表转换为 LLM API 格式的 dict 列表。

        子类可覆写（如 Claude API 需要 system 单独传）。
        """
        result: list[dict[str, object]] = []
        for msg in messages:
            text_contents: list[str] = []
            tool_calls: list[dict[str, object]] = []
            tool_results: list[dict[str, object]] = []

            for part in msg.parts:
                if isinstance(part, TextPart):
                    text_contents.append(part.text)
                elif isinstance(part, ToolPart):
                    state = part.state
                    call_id = part.call_id
                    tool = part.tool
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

            # ReasoningPart 被有意跳过，因为大多数 LLM provider 不支持将 reasoning 内容回传到后续请求中。
            if not text_contents and not tool_calls:
                continue

            role = msg.info.role if msg.info.role != "compaction" else "user"
            main_msg: dict[str, object] = {
                "role": role,
                "content": "".join(text_contents),
            }
            if msg.info.role == "assistant" and tool_calls:
                main_msg["tool_calls"] = tool_calls
            result.append(main_msg)
            result.extend(tool_results)

        return result

    @staticmethod
    def prepare_tools(tools: list[ToolDefinition]) -> list[dict[str, object]]:
        """将 ToolDefinition 列表转换为 OpenAI function-calling 格式。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    @abstractmethod
    def stream_chat(
        self,
        messages: list[InternalMessage],
        tools: list[ToolDefinition] | None = None,
        **kwargs: object,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
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
        tools: list[ToolDefinition] | None = None,
        **kwargs: object,
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
