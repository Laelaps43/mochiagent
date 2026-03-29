from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast, final, override

import pytest

import agent as api
from agent.base_agent import BaseAgent
from agent.core.compression import CompactorRunOptions
from agent.core.runtime import StrategyKind
from agent.core.compression.compactor import NoopContextCompactor
from agent.framework import reset_framework
from agent.types import LLMConfig


def _llm_config() -> LLMConfig:
    return LLMConfig(adapter="openai_compatible", provider="test", model="m1")


@final
class _SimpleAgent(BaseAgent):
    @staticmethod
    @override
    def name() -> str:
        return "simple"

    @staticmethod
    @override
    def description() -> str:
        return "simple agent"

    @property
    @override
    def skill_directory(self) -> Path | None:
        return None

    @property
    @override
    def allowed_model_profiles(self) -> set[str]:
        return {"test:m1"}

    @override
    async def setup(self) -> None:
        return None


@pytest.fixture(autouse=True)
async def reset_framework_and_registry() -> AsyncIterator[None]:
    await reset_framework()
    registered = cast("list[object]", getattr(api, "_registered_agent_classes"))
    registered.clear()
    yield
    await reset_framework()
    registered.clear()


async def test_setup_and_shutdown_with_default_storage() -> None:
    await api.setup(
        agents=[_SimpleAgent()],
        llm_configs=[_llm_config()],
        max_concurrent=4,
        max_iterations=10,
    )
    agent = await api.get_agent("simple")
    assert agent is not None
    assert agent.name() == "simple"
    names = await api.list_agents()
    assert "simple" in names
    await api.shutdown()


async def test_setup_with_explicit_storage() -> None:
    from agent.core.storage import MemoryStorage

    storage = MemoryStorage()
    await api.setup(storage=storage, max_concurrent=4, max_iterations=10)
    assert await api.get_agent("nonexistent") is None
    await api.shutdown()


async def test_set_agent_strategy() -> None:
    await api.setup(
        agents=[_SimpleAgent()],
        llm_configs=[_llm_config()],
        max_concurrent=4,
        max_iterations=10,
    )
    compactor = NoopContextCompactor()
    await api.set_agent_strategy(
        StrategyKind.CONTEXT_COMPACTION,
        "simple",
        compactor,
        compaction_options=CompactorRunOptions(),
    )
    await api.shutdown()


def test_agent_decorator_registers_class() -> None:
    @api.agent
    @final
    class _DecoratedAgent(BaseAgent):
        @staticmethod
        @override
        def name() -> str:
            return "decorated"

        @staticmethod
        @override
        def description() -> str:
            return "decorated agent"

        @property
        @override
        def skill_directory(self) -> Path | None:
            return None

        @property
        @override
        def allowed_model_profiles(self) -> set[str]:
            return set()

        @override
        async def setup(self) -> None:
            return None

    registered = api.get_registered_agents()
    assert _DecoratedAgent in registered
