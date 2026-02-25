import pytest

from agent.core.llm import AdapterRegistry
from agent.types import LLMConfig


def test_unknown_provider_fails_fast_with_available_list():
    registry = AdapterRegistry()
    config = LLMConfig(adapter="unknown-adapter", provider="openai", model="demo")

    with pytest.raises(ValueError) as exc:
        registry.get(config)

    message = str(exc.value)
    assert "unknown-adapter" in message
    assert "Available adapters" in message
