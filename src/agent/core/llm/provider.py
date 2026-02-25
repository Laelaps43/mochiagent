"""
Provider Registry - LLM提供商注册表
管理和创建LLM提供商实例
"""

from typing import Dict, Type

from loguru import logger

from agent.types import LLMConfig
from .base import LLMProvider
from .adapters.openai import OpenAIAdapter


class AdapterRegistry:
    """LLM适配器注册表"""

    def __init__(self):
        self._providers: Dict[str, Type[LLMProvider]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """注册默认适配器"""
        self.register("openai_compatible", OpenAIAdapter)
        logger.info("Registered default LLM adapters (openai_compatible)")

    def register(self, name: str, provider_class: Type[LLMProvider]) -> None:
        """
        注册LLM适配器

        Args:
            name: 适配器名称
            provider_class: 适配器类
        """
        self._providers[name] = provider_class
        logger.debug(f"Registered LLM provider: {name}")

    def get(self, config: LLMConfig) -> LLMProvider:
        """
        获取LLM适配器实例

        Args:
            config: LLM配置

        Returns:
            LLMProvider实例

        Raises:
            ValueError: 如果适配器不存在
        """
        provider_name = config.adapter

        if provider_name not in self._providers:
            available = ", ".join(sorted(self._providers.keys())) or "<none>"
            raise ValueError(
                f"Adapter '{provider_name}' not found. Available adapters: {available}"
            )

        provider_class = self._providers[provider_name]
        return provider_class(config)

    def list_adapters(self) -> list[str]:
        """列出所有已注册的适配器"""
        return list(self._providers.keys())
