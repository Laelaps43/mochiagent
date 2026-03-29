"""SubAgent 基类 - 可被父 Agent 调用的子 Agent"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import override

from .base_agent import BaseAgent
from .config import ToolPolicyConfig
from .core.session.context import SessionContext


class SubAgentBase(BaseAgent, ABC):
    """
    SubAgent 基类，继承 BaseAgent。

    与 BaseAgent 的区别：
    - 额外定义 system_prompt 和 model_profile_id 静态方法
    - 覆盖 get_system_prompt() 直接返回 self.system_prompt()
    - 注册时传类（不传实例），调用时才实例化→setup→执行→cleanup
    """

    @staticmethod
    @abstractmethod
    def system_prompt() -> str:
        """SubAgent 独立的 system prompt。"""
        ...

    @staticmethod
    @abstractmethod
    def model_profile_id() -> str:
        """SubAgent 使用的 model profile（格式：provider:model，如 'openai:gpt-4o-mini'）。"""
        ...

    @staticmethod
    def tool_policy() -> ToolPolicyConfig | None:
        """SubAgent 工具权限策略。返回 None 表示无额外限制。

        Example:
            >>> @staticmethod
            >>> def tool_policy() -> ToolPolicyConfig:
            ...     return ToolPolicyConfig(deny={"exec", "task"})
        """
        return None

    @override
    def get_system_prompt(self, _context: SessionContext) -> str | None:
        return self.system_prompt()
