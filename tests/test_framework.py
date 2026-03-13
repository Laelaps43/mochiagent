from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import final, override

import pytest

from agent.base_agent import BaseAgent
from agent.core.storage import MemoryStorage
from agent.framework import (
    AgentFramework,
    FrameworkRegistry,
    get_framework,
    reset_framework,
)
from agent.types import LLMConfig


def _make_llm_config(
    *,
    adapter: str = "openai_compatible",
    provider: str = "test",
    model: str = "m1",
) -> LLMConfig:
    return LLMConfig(adapter=adapter, provider=provider, model=model)


@final
class _FrameworkAgent(BaseAgent):
    def __init__(
        self,
        *,
        agent_name: str,
        allowed_profiles: set[str],
        setup_error: Exception | None = None,
        cleanup_error: Exception | None = None,
    ) -> None:
        super().__init__()
        self._agent_name: str = agent_name
        self._allowed_profiles: set[str] = allowed_profiles
        self._setup_error: Exception | None = setup_error
        self._cleanup_error: Exception | None = cleanup_error
        self.setup_calls: int = 0
        self.cleanup_calls: int = 0

    @property
    @override
    def name(self) -> str:
        return self._agent_name

    @property
    @override
    def description(self) -> str:
        return f"description for {self._agent_name}"

    @property
    @override
    def skill_directory(self) -> Path | None:
        return None

    @property
    @override
    def allowed_model_profiles(self) -> set[str]:
        return self._allowed_profiles

    @override
    async def setup(self) -> None:
        self.setup_calls += 1
        if self._setup_error is not None:
            raise self._setup_error

    @override
    async def cleanup(self) -> None:
        self.cleanup_calls += 1
        if self._cleanup_error is not None:
            raise self._cleanup_error


@pytest.fixture(autouse=True)
def _reset_framework() -> Iterator[None]:
    reset_framework()
    yield
    reset_framework()


_ = _reset_framework


async def test_framework_init_and_initialize_behaviors() -> None:
    framework = AgentFramework(max_concurrent=10, max_iterations=0)

    assert framework.max_concurrent == 10
    assert framework.max_iterations == 1
    assert framework.is_initialized() is False
    assert framework.is_running() is False
    assert framework.strategy_manager is not None

    await framework.initialize(MemoryStorage())

    assert framework.is_initialized() is True
    assert framework.session_manager is not None
    assert framework.event_loop is not None

    with pytest.raises(RuntimeError, match="already initialized"):
        await framework.initialize(MemoryStorage())


async def test_register_agent_requires_initialized_framework() -> None:
    framework = AgentFramework()

    with pytest.raises(RuntimeError, match="not initialized"):
        await framework.register_agent(
            _FrameworkAgent(agent_name="alpha", allowed_profiles={"test:m1"})
        )


async def test_register_agent_raises_when_session_manager_missing() -> None:
    framework = AgentFramework(max_concurrent=10, max_iterations=10)
    await framework.initialize(MemoryStorage())
    framework.session_manager = None

    with pytest.raises(RuntimeError, match="session_manager is None"):
        await framework.register_agent(
            _FrameworkAgent(agent_name="alpha", allowed_profiles={"test:m1"})
        )


async def test_register_agent_binds_filtered_profiles_and_detects_name_conflict() -> None:
    framework = AgentFramework(max_concurrent=10, max_iterations=10)
    await framework.initialize(MemoryStorage())
    framework.set_llm_configs(
        [
            _make_llm_config(provider="TEST", model="m1"),
            _make_llm_config(provider="test", model="m2"),
        ]
    )
    agent = _FrameworkAgent(agent_name="alpha", allowed_profiles={" test :m1 "})

    await framework.register_agent(agent)

    assert framework.get_agent("alpha") is agent
    assert framework.list_agents() == ["alpha"]
    assert agent.context.agent_name == "alpha"
    assert set(agent.context.llm_profiles) == {"test:m1"}
    assert agent.setup_calls == 1

    with pytest.raises(ValueError, match="already registered"):
        await framework.register_agent(
            _FrameworkAgent(agent_name="alpha", allowed_profiles={"test:m1"})
        )


async def test_register_agent_setup_failure_rolls_back_registry_entry() -> None:
    framework = AgentFramework(max_concurrent=10, max_iterations=10)
    await framework.initialize(MemoryStorage())
    framework.set_llm_configs([_make_llm_config()])
    agent = _FrameworkAgent(
        agent_name="broken",
        allowed_profiles={"test:m1"},
        setup_error=RuntimeError("setup failed"),
    )

    with pytest.raises(RuntimeError, match="setup failed"):
        await framework.register_agent(agent)

    assert framework.get_agent("broken") is None
    assert framework.list_agents() == []


def test_set_llm_configs_validates_adapters_and_conflicts() -> None:
    framework = AgentFramework(max_concurrent=10, max_iterations=10)

    with pytest.raises(ValueError, match="Unknown adapter"):
        framework.set_llm_configs([_make_llm_config(adapter="missing")])

    with pytest.raises(ValueError, match="Conflicting llm config"):
        framework.set_llm_configs(
            [
                _make_llm_config(provider="test", model="m1"),
                _make_llm_config(provider="test", model="m1", adapter="openai_compatible"),
                LLMConfig(
                    adapter="openai_compatible",
                    provider="test",
                    model="m1",
                    temperature=0.1,
                ),
            ]
        )


async def test_unregister_agent_removes_registered_agent() -> None:
    framework = AgentFramework(max_concurrent=10, max_iterations=10)
    await framework.initialize(MemoryStorage())
    framework.set_llm_configs([_make_llm_config()])
    agent = _FrameworkAgent(agent_name="alpha", allowed_profiles={"test:m1"})
    await framework.register_agent(agent)

    framework.unregister_agent("alpha")

    assert framework.get_agent("alpha") is None
    assert framework.list_agents() == []


async def test_start_stop_and_double_start_behaviors() -> None:
    framework = AgentFramework(max_concurrent=10, max_iterations=10)

    with pytest.raises(RuntimeError, match="not initialized"):
        await framework.start()

    await framework.initialize(MemoryStorage())
    framework.set_llm_configs([_make_llm_config()])
    agent = _FrameworkAgent(agent_name="alpha", allowed_profiles={"test:m1"})
    await framework.register_agent(agent)

    await framework.stop()
    assert framework.is_running() is False

    await framework.start()
    assert framework.is_running() is True

    await framework.start()
    assert framework.is_running() is True

    await framework.stop()

    assert framework.is_running() is False
    assert agent.cleanup_calls == 1


async def test_stop_continues_when_agent_cleanup_fails() -> None:
    framework = AgentFramework(max_concurrent=10, max_iterations=10)
    await framework.initialize(MemoryStorage())
    framework.set_llm_configs([_make_llm_config()])
    agent = _FrameworkAgent(
        agent_name="alpha",
        allowed_profiles={"test:m1"},
        cleanup_error=RuntimeError("cleanup failed"),
    )
    await framework.register_agent(agent)
    await framework.start()

    await framework.stop()

    assert framework.is_running() is False
    assert agent.cleanup_calls == 1


async def test_framework_async_context_manager_starts_and_stops() -> None:
    framework = AgentFramework(max_concurrent=10, max_iterations=10)
    await framework.initialize(MemoryStorage())

    async with framework as entered:
        assert entered is framework
        assert framework.is_running() is True

    assert framework.is_running() is False


def test_framework_registry_should_recreate_instance_covers_all_branches() -> None:
    current = AgentFramework(max_concurrent=10, max_iterations=10)

    assert (
        FrameworkRegistry.should_recreate_instance(
            current=current,
            requested_max_concurrent=11,
            requested_max_iterations=None,
            resolved_max_concurrent=11,
            resolved_max_iterations=10,
        )
        is True
    )
    assert (
        FrameworkRegistry.should_recreate_instance(
            current=current,
            requested_max_concurrent=None,
            requested_max_iterations=9,
            resolved_max_concurrent=10,
            resolved_max_iterations=9,
        )
        is True
    )
    assert (
        FrameworkRegistry.should_recreate_instance(
            current=current,
            requested_max_concurrent=10,
            requested_max_iterations=10,
            resolved_max_concurrent=10,
            resolved_max_iterations=10,
        )
        is False
    )
    assert (
        FrameworkRegistry.should_recreate_instance(
            current=current,
            requested_max_concurrent=None,
            requested_max_iterations=None,
            resolved_max_concurrent=50,
            resolved_max_iterations=100,
        )
        is False
    )


def test_framework_registry_get_creates_reuses_and_recreates_uninitialized_instance() -> None:
    registry = FrameworkRegistry()

    first = registry.get()
    second = registry.get()
    recreated = registry.get(max_concurrent=8, max_iterations=0)

    assert second is first
    assert recreated is not first
    assert recreated.max_concurrent == 8
    assert recreated.max_iterations == 1


async def test_framework_registry_get_ignores_new_config_after_initialization() -> None:
    registry = FrameworkRegistry()
    current = registry.get(max_concurrent=10, max_iterations=10)
    await current.initialize(MemoryStorage())

    same = registry.get(max_concurrent=99, max_iterations=3)

    assert same is current
    assert same.max_concurrent == 10
    assert same.max_iterations == 10


def test_get_framework_and_reset_framework_manage_module_singleton() -> None:
    first = get_framework(max_concurrent=7, max_iterations=12)
    second = get_framework()

    assert second is first
    assert first.max_concurrent == 7
    assert first.max_iterations == 12

    reset_framework()

    third = get_framework(max_concurrent=9, max_iterations=4)

    assert third is not first
    assert third.max_concurrent == 9
    assert third.max_iterations == 4
