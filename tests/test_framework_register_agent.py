from __future__ import annotations

from pathlib import Path
from typing import override

import pytest

from agent.base_agent import BaseAgent
from agent.core.storage import MemoryStorage
from agent.framework import AgentFramework, reset_framework


class _MinimalAgent(BaseAgent):
    @property
    @override
    def name(self) -> str:
        return "minimal"

    @property
    @override
    def description(self) -> str:
        return "test agent"

    @property
    @override
    def skill_directory(self) -> Path | None:
        return None

    @property
    @override
    def allowed_model_profiles(self) -> set[str]:
        return {"openai:gpt-4o-mini"}

    @override
    async def setup(self) -> None:
        pass


class _AnotherAgent(BaseAgent):
    @property
    @override
    def name(self) -> str:
        return "another"

    @property
    @override
    def description(self) -> str:
        return "second test agent"

    @property
    @override
    def skill_directory(self) -> Path | None:
        return None

    @property
    @override
    def allowed_model_profiles(self) -> set[str]:
        return {"openai:gpt-4o-mini"}

    @override
    async def setup(self) -> None:
        pass


@pytest.fixture(autouse=True)
def cleanup_framework():
    yield
    reset_framework()


@pytest.fixture
async def initialized_framework() -> AgentFramework:
    fw = AgentFramework()
    await fw.initialize(MemoryStorage())
    return fw


def test_base_agent_without_allowed_model_profiles_is_abstract():
    class _Incomplete(BaseAgent):  # pyright: ignore[reportImplicitAbstractClass]
        @property
        @override
        def name(self) -> str:
            return "incomplete"

        @property
        @override
        def description(self) -> str:
            return "incomplete"

        @property
        @override
        def skill_directory(self) -> Path | None:
            return None

        @override
        async def setup(self) -> None:
            pass

    with pytest.raises(TypeError):
        _ = _Incomplete()  # pyright: ignore[reportAbstractUsage]


async def test_register_agent_before_initialize_raises():
    fw = AgentFramework()
    with pytest.raises(RuntimeError, match="not initialized"):
        await fw.register_agent(_MinimalAgent())


async def test_register_agent_duplicate_raises(initialized_framework: AgentFramework):
    await initialized_framework.register_agent(_MinimalAgent())
    with pytest.raises(ValueError, match="already registered"):
        await initialized_framework.register_agent(_MinimalAgent())


async def test_register_agent_success(initialized_framework: AgentFramework):
    await initialized_framework.register_agent(_MinimalAgent())
    agent = initialized_framework.get_agent("minimal")
    assert agent is not None
    assert agent.name == "minimal"


async def test_register_multiple_agents(initialized_framework: AgentFramework):
    await initialized_framework.register_agent(_MinimalAgent())
    await initialized_framework.register_agent(_AnotherAgent())
    assert initialized_framework.get_agent("minimal") is not None
    assert initialized_framework.get_agent("another") is not None


async def test_get_agent_missing_returns_none(initialized_framework: AgentFramework):
    result = initialized_framework.get_agent("ghost")
    assert result is None


async def test_list_agents_empty_initially(initialized_framework: AgentFramework):
    assert initialized_framework.list_agents() == []


async def test_list_agents_after_registration(initialized_framework: AgentFramework):
    await initialized_framework.register_agent(_MinimalAgent())
    assert "minimal" in initialized_framework.list_agents()


async def test_initialize_twice_raises():
    fw = AgentFramework()
    await fw.initialize(MemoryStorage())
    with pytest.raises(RuntimeError, match="already initialized"):
        await fw.initialize(MemoryStorage())


async def test_framework_is_initialized_flag(initialized_framework: AgentFramework):
    assert initialized_framework.is_initialized() is True


def test_framework_not_initialized_by_default():
    fw = AgentFramework()
    assert fw.is_initialized() is False
