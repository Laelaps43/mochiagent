from __future__ import annotations

import asyncio
from pathlib import Path
from typing import final, override, cast

import pytest

from agent.core.tools.base import Tool
from agent.core.tools.executor import ToolExecutor
from agent.config.tools import ToolPolicyConfig, ToolSecurityConfig
from agent.core.tools.policy import ToolPolicyEngine
from agent.core.tools.registry import ToolRegistry
from agent.types import ToolCallPayload, ToolFunctionPayload


@final
class _EchoTool(Tool):
    @property
    @override
    def name(self) -> str:
        return "echo"

    @property
    @override
    def description(self) -> str:
        return "echo"

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
class _SlowTool(Tool):
    @property
    @override
    def name(self) -> str:
        return "slow"

    @property
    @override
    def description(self) -> str:
        return "slow tool"

    @property
    @override
    def parameters_schema(self) -> dict[str, object]:
        return {"type": "object", "properties": {}}

    @override
    async def execute(self, **kwargs: object) -> object:
        await asyncio.sleep(999)
        return {}


def _make_call(name: str, args: str) -> ToolCallPayload:
    return ToolCallPayload(
        id=f"call_{name}",
        function=ToolFunctionPayload(name=name, arguments=args),
    )


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_EchoTool())
    reg.register(_SlowTool())
    return reg


@pytest.fixture
def executor(registry: ToolRegistry) -> ToolExecutor:
    return ToolExecutor(registry=registry, default_timeout=30)


async def test_execute_success(executor: ToolExecutor):
    call = _make_call("echo", '{"text": "hello"}')
    result = await executor.execute(call)
    assert result.success is True
    assert result.tool_name == "echo"
    assert result.error is None


async def test_execute_invalid_json(executor: ToolExecutor):
    call = _make_call("echo", "not json{{{")
    result = await executor.execute(call)
    assert result.success is False
    assert "Invalid JSON" in (result.error or "")


async def test_execute_tool_not_found(executor: ToolExecutor):
    call = _make_call("nonexistent", "{}")
    result = await executor.execute(call)
    assert result.success is False
    assert "not found" in (result.error or "")


async def test_execute_policy_denied(registry: ToolRegistry):
    engine = ToolPolicyEngine(ToolPolicyConfig(deny={"echo"}))
    exec_ = ToolExecutor(registry=registry, policy=engine)
    call = _make_call("echo", '{"text": "hi"}')
    result = await exec_.execute(call)
    assert result.success is False
    assert "TOOL_POLICY_DENIED" in (result.error or "")


async def test_execute_schema_validation_failure(executor: ToolExecutor):
    call = _make_call("echo", '{"text": 123}')
    result = await executor.execute(call)
    assert result.success is False
    assert "Parameter validation failed" in (result.error or "")


async def test_execute_timeout(registry: ToolRegistry):
    exec_ = ToolExecutor(registry=registry, default_timeout=1)
    call = _make_call("slow", "{}")
    result = await exec_.execute(call)
    assert result.success is False
    assert "timeout" in (result.error or "").lower()


async def test_execute_batch_empty(executor: ToolExecutor):
    results = await executor.execute_batch([])
    assert results == []


async def test_execute_batch_multiple(executor: ToolExecutor):
    calls = [
        _make_call("echo", '{"text": "a"}'),
        _make_call("echo", '{"text": "b"}'),
    ]
    results = await executor.execute_batch(calls)
    assert len(results) == 2
    assert all(r.success for r in results)


async def test_execute_batch_mixed_results(registry: ToolRegistry):
    exec_ = ToolExecutor(registry=registry, default_timeout=30)
    calls = [
        _make_call("echo", '{"text": "ok"}'),
        _make_call("echo", "BAD JSON"),
    ]
    results = await exec_.execute_batch(calls)
    assert len(results) == 2
    assert results[0].success is True
    assert results[1].success is False


async def test_execute_security_guard_blocks(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    @final
    class _PathTool(Tool):
        @property
        @override
        def name(self) -> str:
            return "path_tool"

        @property
        @override
        def description(self) -> str:
            return "path"

        @property
        @override
        def parameters_schema(self) -> dict[str, object]:
            return {
                "type": "object",
                "properties": {"path": {"type": "string", "x-workspace-path": True}},
            }

        @override
        async def execute(self, **kwargs: object) -> object:
            return {}

    reg = ToolRegistry()
    reg.register(_PathTool())
    exec_ = ToolExecutor(
        registry=reg,
        workspace_root=workspace,
        restrict_to_workspace=True,
        security=ToolSecurityConfig(enforce_workspace=True),
    )
    call = _make_call("path_tool", '{"path": "/etc/passwd"}')
    result = await exec_.execute(call)
    assert result.success is False
    assert "TOOL_SECURITY_DENIED" in (result.error or "")


async def test_execute_batch_exception_path(executor: ToolExecutor):
    from unittest.mock import AsyncMock, patch

    calls = [_make_call("echo", '{"text": "a"}')]
    with patch.object(
        executor, "execute", new_callable=AsyncMock, side_effect=RuntimeError("boom")
    ):
        results = await executor.execute_batch(calls)
    assert len(results) == 1
    assert results[0].success is False
    assert "boom" in (results[0].error or "")


def test_tool_to_definition_returns_correct_fields() -> None:
    tool = _EchoTool()
    defn = tool.to_definition()
    assert defn.name == "echo"
    assert defn.description == "echo"
    assert defn.required == ["text"]


async def test_tool_abstract_pass_bodies_return_none() -> None:
    from collections.abc import Callable, Coroutine

    raw_name = cast("Callable[[object], str]", vars(Tool)["name"].fget)
    raw_desc = cast("Callable[[object], str]", vars(Tool)["description"].fget)
    raw_schema = cast("Callable[[object], dict[str, object]]", vars(Tool)["parameters_schema"].fget)
    raw_execute = cast("Callable[..., Coroutine[object, object, object]]", vars(Tool)["execute"])
    tool = _EchoTool()
    assert raw_name(tool) is None
    assert raw_desc(tool) is None
    assert raw_schema(tool) is None
    assert await raw_execute(tool) is None
