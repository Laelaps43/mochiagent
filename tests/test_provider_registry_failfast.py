import pytest

from agent.core.llm import ProviderRegistry
from agent.types import LLMConfig


def test_unknown_provider_fails_fast_with_available_list():
    registry = ProviderRegistry()
    config = LLMConfig(provider="unknown-provider", model="demo")

    with pytest.raises(ValueError) as exc:
        registry.get(config)

    message = str(exc.value)
    assert "unknown-provider" in message
    assert "Available providers" in message
