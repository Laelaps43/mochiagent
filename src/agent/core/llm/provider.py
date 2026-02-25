"""
Provider Registry - LLM提供商注册表
管理和创建LLM提供商实例
"""

from typing import Dict, Type

from loguru import logger

from agent.types import LLMConfig
from .base import LLMProvider
from .adapters.openai import OpenAIAdapter


class ProviderRegistry:
    """LLM提供商注册表"""

    def __init__(self):
        self._providers: Dict[str, Type[LLMProvider]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """注册默认提供商"""
        self.register("openai", OpenAIAdapter)
        logger.info("Registered default LLM providers (openai)")

    def register(self, name: str, provider_class: Type[LLMProvider]) -> None:
        """
        注册LLM提供商

        Args:
            name: 提供商名称
            provider_class: 提供商类
        """
        self._providers[name] = provider_class
        logger.debug(f"Registered LLM provider: {name}")

    def get(self, config: LLMConfig) -> LLMProvider:
        """
        获取LLM提供商实例

        Args:
            config: LLM配置

        Returns:
            LLMProvider实例

        Raises:
            ValueError: 如果提供商不存在
        """
        provider_name = config.provider

        if provider_name not in self._providers:
            available = ", ".join(sorted(self._providers.keys())) or "<none>"
            raise ValueError(
                f"Provider '{provider_name}' not found. Available providers: {available}"
            )

        provider_class = self._providers[provider_name]
        return provider_class(config)

    def list_providers(self) -> list[str]:
        """列出所有已注册的提供商"""
        return list(self._providers.keys())
