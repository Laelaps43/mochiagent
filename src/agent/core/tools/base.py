"""
Tool Base - 工具基类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


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
    def parameters_schema(self) -> Dict[str, Any]:
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

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
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
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=schema,
            required=schema.get("required", []),
        )
