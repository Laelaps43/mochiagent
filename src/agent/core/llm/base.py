"""
LLM Provider Base - LLM提供商基类
定义统一的LLM接口
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

from agent.types import LLMConfig, ToolDefinition


class LLMProvider(ABC):
    """
    LLM提供商基类
    所有LLM适配器必须继承此类
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        流式对话

        Args:
            messages: LLM API 格式的消息列表
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
        messages: List[Dict[str, Any]],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        非流式对话

        Args:
            messages: LLM API 格式的消息列表
            tools: 可用工具列表
            **kwargs: 额外参数

        Returns:
            Dict包含:
            - content: str
            - tool_calls: List[Dict]
            - finish_reason: str
        """
        pass
