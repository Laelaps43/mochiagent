"""Tests for SubAgentBase."""

from __future__ import annotations

from typing import override

import pytest

from agent.config import ToolPolicyConfig, ToolRuntimeConfig
from agent.core.tools.base import Tool
from agent.sub_agent import SubAgentBase


class _EchoTool(Tool):
    @property
    @override
    def name(self) -> str:
        return "echo"

    @property
    @override
    def description(self) -> str:
        return "Echo input"

    @property
    @override
    def parameters_schema(self) -> dict[str, object]:
        return {"type": "object", "properties": {"text": {"type": "string"}}}

    @override
    async def execute(self, text: str = "", **kwargs: object) -> object:
        return text


class _TestSubAgent(SubAgentBase):
    @staticmethod
    @override
    def name() -> str:
        return "test_sub"

    @staticmethod
    @override
    def description() -> str:
        return "A test subagent"

    @staticmethod
    @override
    def system_prompt() -> str:
        return "You are a test subagent."

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
        self.register_tool(_EchoTool())


def test_subagent_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        _ = SubAgentBase()  # pyright: ignore[reportAbstractUsage]


def test_subagent_inherits_base_agent() -> None:
    from agent.base_agent import BaseAgent

    sub = _TestSubAgent()
    assert isinstance(sub, BaseAgent)


def test_subagent_properties() -> None:
    sub = _TestSubAgent()
    assert sub.name() == "test_sub"
    assert sub.description() == "A test subagent"
    assert sub.system_prompt() == "You are a test subagent."
    assert sub.model_profile_id() == "openai:gpt-4o-mini"
    assert sub.allowed_model_profiles == {"openai:gpt-4o-mini"}


@pytest.mark.anyio
async def test_subagent_setup_registers_tools() -> None:
    sub = _TestSubAgent()
    assert not sub.tool_registry.has("echo")
    await sub.setup()
    assert sub.tool_registry.has("echo")


def test_subagent_get_system_prompt_returns_system_prompt() -> None:
    from agent.core.session.context import SessionContext

    sub = _TestSubAgent()
    ctx = SessionContext(session_id="test", model_profile_id="openai:gpt-4o-mini")
    assert sub.get_system_prompt(ctx) == "You are a test subagent."


def test_subagent_with_custom_tool_config() -> None:
    config = ToolRuntimeConfig(policy=ToolPolicyConfig(deny={"exec"}))
    sub = _TestSubAgent(tools=config)
    assert sub.tool_runtime.policy.deny == {"exec"}


@pytest.mark.anyio
async def test_subagent_cleanup() -> None:
    sub = _TestSubAgent()
    await sub.setup()
    assert sub.tool_registry.has("echo")
    await sub.cleanup()
    assert not sub.tool_registry.has("echo")
