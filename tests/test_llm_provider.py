from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import final, override

import pytest
from pydantic import SecretStr

from agent.core.llm.base import LLMProvider
from agent.core.message.message import Message
from agent.core.llm.provider import AdapterRegistry
from agent.types import LLMConfig, LLMStreamChunk, ToolDefinition


def _make_config(adapter: str = "openai_compatible") -> LLMConfig:
    return LLMConfig(adapter=adapter, provider="test", model="m1", api_key=SecretStr("test-key"))


@final
class _CustomProvider(LLMProvider):
    @override
    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: object,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        del messages, tools, kwargs
        yield LLMStreamChunk(content="stream")

    @override
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: object,
    ) -> LLMStreamChunk:
        del messages, tools, kwargs
        return LLMStreamChunk(content="complete")


def test_adapter_registry_registers_default_adapter_on_init() -> None:
    registry = AdapterRegistry()

    assert "openai_compatible" in registry.list_adapters()


def test_adapter_registry_register_and_get_custom_provider() -> None:
    registry = AdapterRegistry()
    registry.register("custom", _CustomProvider)

    provider = registry.get(_make_config(adapter="custom"))

    assert isinstance(provider, _CustomProvider)
    assert provider.config.adapter == "custom"


def test_adapter_registry_get_default_provider_instance() -> None:
    registry = AdapterRegistry()

    provider = registry.get(_make_config())

    assert provider.__class__.__name__ == "OpenAIAdapter"
    assert provider.config.model == "m1"


def test_adapter_registry_get_unknown_adapter_raises() -> None:
    registry = AdapterRegistry()

    with pytest.raises(ValueError, match="Adapter 'missing' not found"):
        _ = registry.get(_make_config(adapter="missing"))


def test_adapter_registry_list_adapters_reflects_registrations() -> None:
    registry = AdapterRegistry()
    registry.register("custom", _CustomProvider)

    adapters = registry.list_adapters()

    assert set(adapters) == {"openai_compatible", "custom"}
