from __future__ import annotations

from typing import final, override

import pytest

from agent.core.tools.base import Tool
from agent.core.tools.registry import ToolRegistry
from agent.types import ToolDefinition


@final
class _EchoTool(Tool):
    @property
    @override
    def name(self) -> str:
        return "echo"

    @property
    @override
    def description(self) -> str:
        return "Echoes input"

    @property
    @override
    def parameters_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    @override
    async def execute(self, **kwargs: object) -> object:
        return {"text": kwargs.get("text")}


@final
class _UpperTool(Tool):
    @property
    @override
    def name(self) -> str:
        return "upper"

    @property
    @override
    def description(self) -> str:
        return "Uppercases input"

    @property
    @override
    def parameters_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    @override
    async def execute(self, **kwargs: object) -> object:
        text = kwargs.get("text", "")
        return {"text": str(text).upper()}


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


def test_register_and_has(registry: ToolRegistry):
    registry.register(_EchoTool())
    assert registry.has("echo") is True
    assert registry.has("missing") is False


def test_get_existing(registry: ToolRegistry):
    tool = _EchoTool()
    registry.register(tool)
    assert registry.get("echo") is tool


def test_get_missing_raises(registry: ToolRegistry):
    with pytest.raises(ValueError, match="not found"):
        _ = registry.get("nonexistent")


def test_unregister_existing(registry: ToolRegistry):
    registry.register(_EchoTool())
    registry.unregister("echo")
    assert registry.has("echo") is False


def test_unregister_missing_no_error(registry: ToolRegistry):
    registry.unregister("ghost")


def test_list_tools_empty(registry: ToolRegistry):
    assert registry.list_tools() == []


def test_list_tools(registry: ToolRegistry):
    registry.register(_EchoTool())
    registry.register(_UpperTool())
    names = registry.list_tools()
    assert set(names) == {"echo", "upper"}


def test_get_definitions_returns_tool_definitions(registry: ToolRegistry):
    registry.register(_EchoTool())
    defs = registry.get_definitions()
    assert len(defs) == 1
    assert isinstance(defs[0], ToolDefinition)
    assert defs[0].name == "echo"
    assert defs[0].required == ["text"]


def test_clear(registry: ToolRegistry):
    registry.register(_EchoTool())
    registry.register(_UpperTool())
    registry.clear()
    assert registry.list_tools() == []


def test_register_overwrites(registry: ToolRegistry):
    registry.register(_EchoTool())
    registry.register(_EchoTool())
    assert len(registry.list_tools()) == 1


def test_to_definition():
    tool = _EchoTool()
    defn = tool.to_definition()
    assert defn.name == "echo"
    assert defn.description == "Echoes input"
    props = defn.parameters.get("properties")
    assert isinstance(props, dict)
    assert "text" in props
