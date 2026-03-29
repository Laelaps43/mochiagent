"""
Tool Base - 工具基类
"""

from abc import ABC, abstractmethod
from typing import cast

from agent.types import ToolDefinition


class Tool(ABC):
    """
    工具基类
    所有工具必须继承此类
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, object]:
        """
        参数JSON Schema

        返回格式:
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
        """
        pass

    @property
    def timeout(self) -> int | None:
        """工具自定义超时（秒）。返回 None 使用 executor 默认值。"""
        return None

    @abstractmethod
    async def execute(self, **kwargs: object) -> object:
        """
        执行工具

        Args:
            **kwargs: 工具参数

        Returns:
            执行结果
        """
        pass

    def to_definition(self) -> ToolDefinition:
        """
        转换为工具定义

        Returns:
            ToolDefinition对象
        """
        schema = self.parameters_schema
        raw_required = schema.get("required")
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=schema,
            required=[r for r in cast(list[object], raw_required) if isinstance(r, str)]
            if isinstance(raw_required, list)
            else [],
        )
