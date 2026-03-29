"""Tests for SubAgentRunner."""

from __future__ import annotations

from typing import override

import pytest

from agent.core.bus.message_bus import MessageBus
from agent.core.loop.sub_agent_runner import SubAgentRunner
from agent.core.session.manager import SessionManager
from agent.core.storage import MemoryStorage
from agent.sub_agent import SubAgentBase
from agent.types import LLMConfig


class _DummySubAgent(SubAgentBase):
    @staticmethod
    @override
    def name() -> str:
        return "test_sub"

    @staticmethod
    @override
    def description() -> str:
        return "test"

    @staticmethod
    @override
    def system_prompt() -> str:
        return "You are a test agent."

    @staticmethod
    @override
    def model_profile_id() -> str:
        return "openai:gpt-4o-mini"

    @property
    @override
    def allowed_model_profiles(self) -> set[str]:
        return {"openai:gpt-4o-mini"}

    @override
    async def setup(self) -> None:
        pass


def _make_llm_config() -> LLMConfig:
    return LLMConfig(
        adapter="openai_compatible",
        provider="openai",
        model="gpt-4o-mini",
    )


def _make_runner(
    bus: MessageBus | None = None,
    max_depth: int = 3,
) -> SubAgentRunner:
    config = _make_llm_config()
    resolved_bus = bus or MessageBus()
    storage = MemoryStorage()
    session_manager = SessionManager(bus=resolved_bus, storage=storage)
    from agent.core.llm.provider import AdapterRegistry

    return SubAgentRunner(
        adapter_registry=AdapterRegistry(),
        session_manager=session_manager,
        message_bus=resolved_bus,
        llm_profiles={"openai:gpt-4o-mini": config},
        max_depth=max_depth,
    )


@pytest.mark.anyio
async def test_depth_exceeded_returns_error() -> None:
    runner = _make_runner(max_depth=2)
    result = await runner.run(_DummySubAgent, "hello", parent_session_id="p1", current_depth=3)
    assert not result.success
    assert "depth exceeded" in (result.error or "").lower()
    assert result.session_id == ""


@pytest.mark.anyio
async def test_invalid_profile_falls_back_to_parent() -> None:
    """When subagent's profile is not found, falls back to parent's profile."""

    class _BadProfileSubAgent(_DummySubAgent):
        @staticmethod
        @override
        def model_profile_id() -> str:
            return "unknown:model"

        @property
        @override
        def allowed_model_profiles(self) -> set[str]:
            return {"unknown:model"}

    runner = _make_runner()
    # Fallback 到父 agent 的 profile，但因为没有 API key 会失败在 LLM 调用阶段
    result = await runner.run(_BadProfileSubAgent, "hello", parent_session_id="p1")
    assert not result.success
    assert "not available" not in (result.error or "").lower()
