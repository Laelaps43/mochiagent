"""Tests for TaskTool."""

from __future__ import annotations

from typing import override

import pytest

from agent.common.tools.task_tool import TaskTool
from agent.core.bus.message_bus import MessageBus
from agent.core.session.manager import SessionManager
from agent.core.storage import MemoryStorage
from agent.sub_agent import SubAgentBase
from agent.types import LLMConfig


class _DummySubAgent(SubAgentBase):
    @staticmethod
    @override
    def name() -> str:
        return "dummy"

    @staticmethod
    @override
    def description() -> str:
        return "A dummy sub agent"

    @staticmethod
    @override
    def system_prompt() -> str:
        return "You are dummy."

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


def _make_tool(classes: dict[str, type] | None = None) -> TaskTool:
    from agent.core.llm.provider import AdapterRegistry

    bus = MessageBus()
    return TaskTool(
        subagent_classes=classes or {},
        adapter_registry=AdapterRegistry(),
        session_manager=SessionManager(bus=bus, storage=MemoryStorage()),
        message_bus=bus,
        llm_profiles={
            "openai:gpt-4o-mini": LLMConfig(
                adapter="openai_compatible",
                provider="openai",
                model="gpt-4o-mini",
            )
        },
    )


def test_task_tool_name() -> None:
    assert _make_tool().name == "task"


def test_task_tool_description_no_subagents() -> None:
    assert _make_tool().description == "No sub-agents available."


def test_task_tool_description_lists_subagents() -> None:
    tool = _make_tool({"dummy": _DummySubAgent})
    desc = tool.description
    assert "<name>dummy</name>" in desc
    assert "<description>A dummy sub agent</description>" in desc


def test_task_tool_parameters_schema() -> None:
    schema = _make_tool().parameters_schema
    props = schema["properties"]
    assert isinstance(props, dict)
    assert "agent_name" in props
    assert "prompt" in props


@pytest.mark.anyio
async def test_task_tool_execute_unknown_agent() -> None:
    result = await _make_tool().execute(agent_name="nonexistent", prompt="hello")
    assert isinstance(result, str)
    assert "not found" in result
