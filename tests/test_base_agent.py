from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Callable, final, override, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.base_agent import BaseAgent
from agent.core.mcp import MCPManager
from agent.common.tools import SkillTool
from agent.context import AgentContext
from agent.core.bus import MessageBus
from agent.core.session import SessionManager
from agent.core.session.context import SessionContext
from agent.core.runtime import AgentStrategyManager
from agent.core.storage import MemoryStorage
from agent.core.tools import Tool
from agent.framework import AgentFramework, reset_framework
from agent.types import Event, EventType, LLMConfig


def _make_llm_config() -> LLMConfig:
    return LLMConfig(adapter="openai_compatible", provider="test", model="m1")


@final
class _FakeTool(Tool):
    @property
    @override
    def name(self) -> str:
        return "fake_tool"

    @property
    @override
    def description(self) -> str:
        return "A fake tool"

    @property
    @override
    def parameters_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }

    @override
    async def execute(self, **kwargs: object) -> object:
        return kwargs


@final
class _ConcreteAgent(BaseAgent):
    def __init__(self, skill_dir: Path | None = None) -> None:
        super().__init__()
        self._skill_dir: Path | None = skill_dir

    @property
    @override
    def name(self) -> str:
        return "concrete"

    @property
    @override
    def description(self) -> str:
        return "Concrete test agent"

    @property
    @override
    def skill_directory(self) -> Path | None:
        return self._skill_dir

    @property
    @override
    def allowed_model_profiles(self) -> set[str]:
        return {"test:m1"}

    @override
    async def setup(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _reset_framework() -> Iterator[None]:
    reset_framework()
    yield
    reset_framework()


_ = _reset_framework


def _build_context() -> AgentContext:
    storage = MemoryStorage()
    bus = MessageBus(max_concurrent=4)
    session_manager = SessionManager(bus=bus, storage=storage)
    return AgentContext(
        session_manager=session_manager,
        message_bus=bus,
        strategy_manager=AgentStrategyManager(),
        agent_name="concrete",
        llm_profiles={"test:m1": _make_llm_config()},
    )


async def _build_registered_agent() -> tuple[AgentFramework, _ConcreteAgent]:
    framework = AgentFramework(max_concurrent=10, max_iterations=10)
    await framework.initialize(MemoryStorage())
    framework.set_llm_configs([_make_llm_config()])
    agent = _ConcreteAgent()
    await framework.register_agent(agent)
    return framework, agent


async def test_base_agent_abstract_members_are_callable() -> None:
    agent = _ConcreteAgent()

    name_prop = cast(property, BaseAgent.__dict__["name"])
    description_prop = cast(property, BaseAgent.__dict__["description"])
    skill_directory_prop = cast(property, BaseAgent.__dict__["skill_directory"])
    allowed_profiles_prop = cast(property, BaseAgent.__dict__["allowed_model_profiles"])
    name_getter = cast(Callable[[_ConcreteAgent], str | None], name_prop.fget)
    description_getter = cast(Callable[[_ConcreteAgent], str | None], description_prop.fget)
    skill_directory_getter = cast(
        Callable[[_ConcreteAgent], Path | None], skill_directory_prop.fget
    )
    allowed_profiles_getter = cast(
        Callable[[_ConcreteAgent], set[str] | None], allowed_profiles_prop.fget
    )

    assert isinstance(name_prop, property)
    assert isinstance(description_prop, property)
    assert isinstance(skill_directory_prop, property)
    assert isinstance(allowed_profiles_prop, property)
    assert name_prop.fget is not None
    assert description_prop.fget is not None
    assert skill_directory_prop.fget is not None
    assert allowed_profiles_prop.fget is not None

    assert name_getter(agent) is None
    assert description_getter(agent) is None
    assert skill_directory_getter(agent) is None
    assert allowed_profiles_getter(agent) is None
    assert await BaseAgent.setup(agent) is None


def test_context_property_requires_binding_and_bind_context_works() -> None:
    agent = _ConcreteAgent()

    with pytest.raises(RuntimeError, match="context not bound yet"):
        _ = agent.context

    context = _build_context()
    agent.bind_context(context)

    assert agent.context is context


def test_default_optional_methods_and_properties_return_none_or_empty() -> None:
    agent = _ConcreteAgent()
    session_context = SessionContext(session_id="session-1", model_profile_id="test:m1")

    assert agent.get_system_prompt(session_context) is None
    assert agent.mcp_config_path is None
    assert agent.default_model_profile is None
    assert agent.get_mcp_status() == {}


def test_register_tool_adds_tool_to_registry() -> None:
    agent = _ConcreteAgent()
    tool = _FakeTool()

    agent.register_tool(tool)

    assert agent.tool_registry.has("fake_tool") is True
    assert agent.tool_registry.get("fake_tool") is tool


async def test_register_mcp_tools_returns_early_without_config_and_cleanup_without_manager() -> (
    None
):
    agent = _ConcreteAgent()

    await agent.register_mcp_tools()
    await agent.cleanup()

    assert agent.get_mcp_status() == {}


def test_register_skill_requires_skill_directory() -> None:
    agent = _ConcreteAgent(skill_dir=None)

    with pytest.raises(ValueError, match="has no skill_directory"):
        agent.register_skill("missing")


def test_register_skill_raises_when_skill_missing(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()
    agent = _ConcreteAgent(skill_dir=skill_dir)

    with pytest.raises(ValueError, match="Failed to load skill"):
        agent.register_skill("ghost-skill")


async def test_register_skill_updates_single_skill_tool_on_re_registration(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills"
    first_skill = skill_dir / "first-skill"
    second_skill = skill_dir / "second-skill"
    first_skill.mkdir(parents=True)
    second_skill.mkdir(parents=True)
    _ = (first_skill / "SKILL.md").write_text(
        "---\nname: first-skill\ndescription: first desc\n---\n# First\n", encoding="utf-8"
    )
    _ = (second_skill / "SKILL.md").write_text(
        "---\nname: second-skill\ndescription: second desc\n---\nUse $ARGUMENTS here.\n",
        encoding="utf-8",
    )
    agent = _ConcreteAgent(skill_dir=skill_dir)

    agent.register_skill("first-skill")

    first_tool = agent.tool_registry.get("skill")
    assert isinstance(first_tool, SkillTool)
    assert agent.tool_registry.list_tools() == ["skill"]
    assert "first-skill" in first_tool.description

    agent.register_skill("second-skill")

    second_tool = agent.tool_registry.get("skill")
    assert isinstance(second_tool, SkillTool)
    assert second_tool is not first_tool
    assert agent.tool_registry.list_tools() == ["skill"]
    assert "first-skill" in second_tool.description
    assert "second-skill" in second_tool.description

    rendered = await second_tool.execute(name="second-skill", context="extra context")
    assert isinstance(rendered, str)
    assert "extra context" in rendered


async def test_push_message_requires_bound_context() -> None:
    agent = _ConcreteAgent()

    with pytest.raises(RuntimeError, match="context not bound"):
        await agent.push_message("session-1", "hello")


async def test_push_message_with_string_builds_user_input_and_saves_message() -> None:
    framework, agent = await _build_registered_agent()
    assert framework.session_manager is not None
    _ = await framework.session_manager.get_or_create_session(
        session_id="session-1",
        model_profile_id="test:m1",
        agent_name=agent.name,
    )

    await agent.push_message("session-1", "hello world")

    context = await framework.session_manager.get_session("session-1")
    assert len(context.messages) == 1
    first_part = context.messages[0].parts[0]
    assert first_part.type == "text"
    assert getattr(first_part, "text") == "hello world"


async def test_take_session_requires_context_and_model_profile() -> None:
    agent = _ConcreteAgent()

    with pytest.raises(RuntimeError, match="context not bound"):
        _ = await agent.take_session("session-1", model_profile_id="test:m1")

    framework, bound_agent = await _build_registered_agent()
    assert framework.session_manager is not None

    with pytest.raises(ValueError, match="model_profile_id is required"):
        _ = await bound_agent.take_session("session-2")


async def test_take_session_returns_session_and_switches_agent() -> None:
    framework, agent = await _build_registered_agent()
    assert framework.session_manager is not None
    _ = await framework.session_manager.get_or_create_session(
        session_id="session-3",
        model_profile_id="test:m1",
        agent_name="other-agent",
    )

    session = await agent.take_session("session-3", model_profile_id="test:m1")

    assert session.session_id == "session-3"
    assert session.agent_name == agent.name
    assert session.model_profile_id == "test:m1"

    context = await framework.session_manager.get_session("session-3")
    assert context.agent_name == agent.name


async def test_handle_event_is_default_noop() -> None:
    agent = _ConcreteAgent()
    event = Event(type=EventType.MESSAGE_DONE, session_id="session-1")

    assert await agent.handle_event(event) is None


async def test_register_mcp_tools_connect_exception_calls_close_and_returns(
    tmp_path: Path,
) -> None:
    mcp_json = tmp_path / "mcp.json"
    _ = mcp_json.write_text('{"mcpServers": {"srv": {"command": "echo"}}}', encoding="utf-8")
    agent = _ConcreteAgent()

    with patch(
        "agent.base_agent.MCPManager.connect_servers",
        new=AsyncMock(side_effect=RuntimeError("conn failed")),
    ):
        with patch("agent.base_agent.MCPManager.close", new=AsyncMock()) as mock_close:
            await agent.register_mcp_tools(path=mcp_json)

    mock_close.assert_called_once()
    assert agent.get_mcp_status() == {}


async def test_register_mcp_tools_zero_registered_closes_manager(
    tmp_path: Path,
) -> None:
    mcp_json = tmp_path / "mcp.json"
    _ = mcp_json.write_text('{"mcpServers": {"srv": {"command": "echo"}}}', encoding="utf-8")
    agent = _ConcreteAgent()

    with patch(
        "agent.base_agent.MCPManager.connect_servers",
        new=AsyncMock(return_value=0),
    ):
        with patch("agent.base_agent.MCPManager.close", new=AsyncMock()) as mock_close:
            with patch(
                "agent.base_agent.MCPManager.snapshot",
                return_value={},
            ):
                await agent.register_mcp_tools(path=mcp_json)

    mock_close.assert_called_once()
    assert agent.get_mcp_status() == {}


async def test_register_mcp_tools_success_sets_manager_and_status(
    tmp_path: Path,
) -> None:
    mcp_json = tmp_path / "mcp.json"
    _ = mcp_json.write_text('{"mcpServers": {"srv": {"command": "echo"}}}', encoding="utf-8")
    agent = _ConcreteAgent()

    with patch(
        "agent.base_agent.MCPManager.connect_servers",
        new=AsyncMock(return_value=1),
    ):
        with patch(
            "agent.base_agent.MCPManager.snapshot",
            return_value={"srv": MagicMock()},
        ):
            await agent.register_mcp_tools(path=mcp_json)

    mgr = cast("MCPManager | None", getattr(agent, "_mcp_manager"))
    assert mgr is not None
    assert isinstance(agent.get_mcp_status(), dict)


async def test_cleanup_with_mcp_manager_closes_and_clears_it(
    tmp_path: Path,
) -> None:
    mcp_json = tmp_path / "mcp.json"
    _ = mcp_json.write_text('{"mcpServers": {"srv": {"command": "echo"}}}', encoding="utf-8")
    agent = _ConcreteAgent()

    with patch(
        "agent.base_agent.MCPManager.connect_servers",
        new=AsyncMock(return_value=1),
    ):
        with patch("agent.base_agent.MCPManager.snapshot", return_value={}):
            await agent.register_mcp_tools(path=mcp_json)

    assert cast("MCPManager | None", getattr(agent, "_mcp_manager")) is not None

    with patch("agent.base_agent.MCPManager.close", new=AsyncMock()) as mock_close:
        await agent.cleanup()

    mock_close.assert_called_once()
    assert cast("MCPManager | None", getattr(agent, "_mcp_manager")) is None
