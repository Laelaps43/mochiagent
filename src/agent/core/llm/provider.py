"""
Provider Registry - LLM提供商注册表
管理和创建LLM提供商实例
"""

import hashlib

from loguru import logger

from agent.types import LLMConfig
from .base import LLMProvider
from .adapters.openai import OpenAIAdapter


class AdapterRegistry:
    """LLM适配器注册表"""

    def __init__(self):
        self._providers: dict[str, type[LLMProvider]] = {}
        self._cache: dict[str, LLMProvider] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """注册默认适配器"""
        self.register("openai_compatible", OpenAIAdapter)
        logger.info("Registered default LLM adapters (openai_compatible)")

    def register(self, name: str, provider_class: type[LLMProvider]) -> None:
        """
        注册LLM适配器

        Args:
            name: 适配器名称
            provider_class: 适配器类
        """
        self._providers[name] = provider_class
        logger.debug("Registered LLM provider: {}", name)

    @staticmethod
    def _cache_key(config: LLMConfig) -> str:
        if config.api_key:
            api_key_hash = hashlib.sha256(config.api_key.get_secret_value().encode()).hexdigest()[
                :32
            ]
        else:
            api_key_hash = ""
        return f"{config.adapter}:{config.model}:{config.base_url}:{api_key_hash}"

    def get(self, config: LLMConfig) -> LLMProvider:
        """
        获取LLM适配器实例（相同配置复用缓存实例）

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

        key = self._cache_key(config)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        provider_class = self._providers[provider_name]
        instance = provider_class(config)
        self._cache[key] = instance
        return instance

    def clear_cache(self) -> None:
        """清空适配器实例缓存"""
        self._cache.clear()

    def list_adapters(self) -> list[str]:
        """列出所有已注册的适配器"""
        return list(self._providers.keys())
