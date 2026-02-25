from pathlib import Path
from typing import Any, Dict

import pytest

from agent.core.tools import Tool, ToolExecutor, ToolRegistry


class EchoTool(Tool):
    @property
    def name(self) -> str:
        return "echo_tool"

    @property
    def description(self) -> str:
        return "Echo input"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        }

    async def execute(self, text: str) -> Any:
        return {"text": text}


class ReadFileLikeTool(Tool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read file-like tool for guard testing"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "x-workspace-path": True},
            },
            "required": ["path"],
        }

    async def execute(self, path: str) -> Any:
        return {"path": path}


class ExecLikeTool(Tool):
    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Exec-like tool for guard testing"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "x-shell-command": True},
                "workdir": {"type": "string", "x-workspace-cwd": True},
            },
            "required": ["command"],
        }

    async def execute(self, command: str, workdir: str | None = None) -> Any:
        return {"command": command, "workdir": workdir}


def _tool_call(name: str, args_json: str, call_id: str = "call_1") -> Dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": args_json,
        },
    }


@pytest.mark.asyncio
async def test_policy_allow_empty_defaults_to_allow():
    registry = ToolRegistry()
    registry.register(EchoTool())
    executor = ToolExecutor(
        registry,
        default_timeout=5,
        policy_allow=set(),
        policy_deny=set(),
    )

    result = await executor.execute(_tool_call("echo_tool", '{"text":"ok"}'))
    assert result.success is True
    assert result.result == {"text": "ok"}


@pytest.mark.asyncio
async def test_policy_deny_overrides_allow():
    registry = ToolRegistry()
    registry.register(EchoTool())
    executor = ToolExecutor(
        registry,
        default_timeout=5,
        policy_allow={"echo_tool"},
        policy_deny={"echo_tool"},
    )

    result = await executor.execute(_tool_call("echo_tool", '{"text":"blocked"}'))
    assert result.success is False
    assert result.error is not None
    assert result.error.startswith("TOOL_POLICY_DENIED:")


@pytest.mark.asyncio
async def test_policy_allowlist_blocks_unlisted_tool():
    registry = ToolRegistry()
    registry.register(EchoTool())
    executor = ToolExecutor(
        registry,
        default_timeout=5,
        policy_allow={"some_other_tool"},
        policy_deny=set(),
    )

    result = await executor.execute(_tool_call("echo_tool", '{"text":"blocked"}'))
    assert result.success is False
    assert result.error is not None
    assert result.error.startswith("TOOL_POLICY_DENIED:")


@pytest.mark.asyncio
async def test_workspace_guard_blocks_outside_path(tmp_path: Path):
    outside = tmp_path.parent / "outside.txt"

    registry = ToolRegistry()
    registry.register(ReadFileLikeTool())
    executor = ToolExecutor(
        registry,
        default_timeout=5,
        workspace_root=tmp_path,
        restrict_to_workspace=True,
    )

    result = await executor.execute(_tool_call("read_file", f'{{"path":"{outside.as_posix()}"}}'))
    assert result.success is False
    assert result.error is not None
    assert result.error.startswith("TOOL_SECURITY_DENIED:")


@pytest.mark.asyncio
async def test_workspace_guard_allows_inside_relative_path(tmp_path: Path):
    inside_file = tmp_path / "notes.txt"
    inside_file.write_text("hello", encoding="utf-8")

    registry = ToolRegistry()
    registry.register(ReadFileLikeTool())
    executor = ToolExecutor(
        registry,
        default_timeout=5,
        workspace_root=tmp_path,
        restrict_to_workspace=True,
    )

    result = await executor.execute(_tool_call("read_file", '{"path":"notes.txt"}'))
    assert result.success is True


@pytest.mark.asyncio
async def test_workspace_guard_blocks_exec_workdir_outside(tmp_path: Path):
    outside_dir = tmp_path.parent

    registry = ToolRegistry()
    registry.register(ExecLikeTool())
    executor = ToolExecutor(
        registry,
        default_timeout=5,
        workspace_root=tmp_path,
        restrict_to_workspace=True,
    )

    result = await executor.execute(
        _tool_call(
            "exec",
            f'{{"command":"ls","workdir":"{outside_dir.as_posix()}"}}',
        )
    )
    assert result.success is False
    assert result.error is not None
    assert result.error.startswith("TOOL_SECURITY_DENIED:")


@pytest.mark.asyncio
async def test_security_guard_blocks_exec_absolute_path_outside_workspace(tmp_path: Path):
    outside_file = tmp_path.parent / "outside.txt"

    registry = ToolRegistry()
    registry.register(ExecLikeTool())
    executor = ToolExecutor(
        registry,
        default_timeout=5,
        workspace_root=tmp_path,
        restrict_to_workspace=True,
    )

    result = await executor.execute(
        _tool_call("exec", f'{{"command":"cat {outside_file.as_posix()}"}}')
    )
    assert result.success is False
    assert result.error is not None
    assert result.error.startswith("TOOL_SECURITY_DENIED:")


@pytest.mark.asyncio
async def test_security_guard_blocks_exec_path_traversal(tmp_path: Path):
    registry = ToolRegistry()
    registry.register(ExecLikeTool())
    executor = ToolExecutor(
        registry,
        default_timeout=5,
        workspace_root=tmp_path,
        restrict_to_workspace=True,
    )

    result = await executor.execute(_tool_call("exec", '{"command":"cat ../secret.txt"}'))
    assert result.success is False
    assert result.error is not None
    assert result.error.startswith("TOOL_SECURITY_DENIED:")
