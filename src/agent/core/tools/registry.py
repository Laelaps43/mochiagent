"""
Tool Registry - 工具注册表
"""

from loguru import logger

from .base import Tool
from agent.types import ToolDefinition


class ToolRegistry:
    """
    工具注册表
    管理所有可用的工具
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._definitions_cache: list[ToolDefinition] | None = None
        logger.info("ToolRegistry initialized")

    def _invalidate_cache(self) -> None:
        self._definitions_cache = None

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._invalidate_cache()
        logger.info("Registered tool: {}", tool.name)

    def unregister(self, tool_name: str) -> None:
        if tool_name in self._tools:
            del self._tools[tool_name]
            self._invalidate_cache()
            logger.info("Unregistered tool: {}", tool_name)

    def get(self, tool_name: str) -> Tool:
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        return self._tools[tool_name]

    def has(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_definitions(self) -> list[ToolDefinition]:
        if self._definitions_cache is None:
            self._definitions_cache = [tool.to_definition() for tool in self._tools.values()]
        return self._definitions_cache

    def clear(self) -> None:
        self._tools.clear()
        self._invalidate_cache()
        logger.info("All tools cleared")
