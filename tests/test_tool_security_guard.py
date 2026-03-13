from __future__ import annotations

from pathlib import Path
from typing import final

import pytest

from agent.core.tools.security_guard import ToolSecurityConfig, ToolSecurityGuard


@final
class _FakeTool:
    parameters_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "x-workspace-path": True},
        },
    }


@final
class _FakeCommandTool:
    parameters_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "x-shell-command": True},
        },
    }


@final
class _FakeCommandWithWorkdirTool:
    parameters_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "x-shell-command": True},
            "workdir": {"type": "string", "x-workspace-cwd": True},
        },
    }


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def guard(workspace: Path) -> ToolSecurityGuard:
    return ToolSecurityGuard(
        root=workspace,
        restrict=True,
        config=ToolSecurityConfig(enforce_workspace=True, enforce_command_guard=True),
    )


def test_path_inside_workspace_allowed(guard: ToolSecurityGuard, workspace: Path):
    tool = _FakeTool()
    decision = guard.validate_tool_call(tool, {"path": str(workspace / "file.txt")})
    assert decision.allowed is True


def test_path_outside_workspace_denied(guard: ToolSecurityGuard, tmp_path: Path):
    tool = _FakeTool()
    outside = tmp_path / "outside.txt"
    decision = guard.validate_tool_call(tool, {"path": str(outside)})
    assert decision.allowed is False
    assert "outside workspace root" in decision.reason


def test_restrict_false_skips_path_check(workspace: Path, tmp_path: Path):
    guard = ToolSecurityGuard(
        root=workspace,
        restrict=False,
        config=ToolSecurityConfig(enforce_workspace=True, enforce_command_guard=True),
    )
    tool = _FakeTool()
    outside = tmp_path / "outside.txt"
    decision = guard.validate_tool_call(tool, {"path": str(outside)})
    assert decision.allowed is True


def test_enforce_workspace_false_skips_path_check(workspace: Path, tmp_path: Path):
    guard = ToolSecurityGuard(
        root=workspace,
        restrict=True,
        config=ToolSecurityConfig(enforce_workspace=False, enforce_command_guard=True),
    )
    tool = _FakeTool()
    outside = tmp_path / "outside.txt"
    decision = guard.validate_tool_call(tool, {"path": str(outside)})
    assert decision.allowed is True


def test_command_deny_token_blocks(workspace: Path):
    guard2 = ToolSecurityGuard(
        root=workspace,
        restrict=True,
        config=ToolSecurityConfig(
            enforce_workspace=True,
            enforce_command_guard=True,
            command_deny_tokens={"$("},
        ),
    )
    tool = _FakeCommandTool()
    decision = guard2.validate_tool_call(tool, {"command": "echo $(cat /etc/passwd)"})
    assert decision.allowed is False
    assert "denied token" in decision.reason


def test_command_without_deny_tokens_allowed(workspace: Path):
    guard = ToolSecurityGuard(
        root=workspace,
        restrict=True,
        config=ToolSecurityConfig(
            enforce_workspace=True,
            enforce_command_guard=True,
            command_deny_tokens={"$(", "`"},
        ),
    )
    tool = _FakeCommandTool()
    decision = guard.validate_tool_call(tool, {"command": "ls -la"})
    assert decision.allowed is True


def test_command_guard_disabled_skips_all_checks(workspace: Path):
    guard = ToolSecurityGuard(
        root=workspace,
        restrict=True,
        config=ToolSecurityConfig(
            enforce_workspace=True,
            enforce_command_guard=False,
            command_deny_tokens={"$("},
        ),
    )
    tool = _FakeCommandTool()
    decision = guard.validate_tool_call(tool, {"command": "echo $(bad stuff)"})
    assert decision.allowed is True


def test_workdir_outside_workspace_denied(guard: ToolSecurityGuard, tmp_path: Path):
    tool = _FakeCommandWithWorkdirTool()
    outside_dir = tmp_path / "danger"
    outside_dir.mkdir()
    decision = guard.validate_tool_call(tool, {"command": "ls", "workdir": str(outside_dir)})
    assert decision.allowed is False
    assert "workdir" in decision.reason


def test_command_absolute_path_outside_workspace_denied(guard: ToolSecurityGuard, tmp_path: Path):
    tool = _FakeCommandTool()
    outside_file = str(tmp_path / "etc" / "passwd")
    decision = guard.validate_tool_call(tool, {"command": f"cat {outside_file}"})
    assert decision.allowed is False
    assert "outside workspace root" in decision.reason


def test_malformed_command_denied(guard: ToolSecurityGuard):
    tool = _FakeCommandTool()
    decision = guard.validate_tool_call(tool, {"command": "echo 'unbalanced"})
    assert decision.allowed is False


def test_no_path_argument_passes(guard: ToolSecurityGuard):
    tool = _FakeTool()
    decision = guard.validate_tool_call(tool, {})
    assert decision.allowed is True


def test_tool_with_no_schema_always_allowed(workspace: Path):
    guard = ToolSecurityGuard(
        root=workspace,
        restrict=True,
        config=ToolSecurityConfig(),
    )

    @final
    class _NoSchema:
        parameters_schema: dict[str, object] = {}

    decision = guard.validate_tool_call(_NoSchema(), {"path": "/etc/passwd"})
    assert decision.allowed is True


def test_non_dict_property_value_skipped(workspace: Path):
    guard = ToolSecurityGuard(
        root=workspace,
        restrict=True,
        config=ToolSecurityConfig(enforce_workspace=True),
    )

    @final
    class _WeirdSchemaTool:
        parameters_schema: dict[str, object] = {
            "type": "object",
            "properties": {"path": 42},
        }

    decision = guard.validate_tool_call(_WeirdSchemaTool(), {"path": "/etc/passwd"})
    assert decision.allowed is True


def test_restrict_false_shell_command_skips_workspace_check(workspace: Path, tmp_path: Path):
    guard = ToolSecurityGuard(
        root=workspace,
        restrict=False,
        config=ToolSecurityConfig(enforce_workspace=True, enforce_command_guard=True),
    )
    tool = _FakeCommandTool()
    outside = str(tmp_path / "outside" / "file.txt")
    decision = guard.validate_tool_call(tool, {"command": f"cat {outside}"})
    assert decision.allowed is True


def test_relative_slash_token_inside_workspace_allowed(guard: ToolSecurityGuard, workspace: Path):
    tool = _FakeCommandTool()
    (workspace / "subdir").mkdir()
    decision = guard.validate_tool_call(tool, {"command": "cat subdir/file.txt"})
    assert decision.allowed is True


def test_home_path_outside_workspace_denied(guard: ToolSecurityGuard):
    tool = _FakeCommandTool()
    decision = guard.validate_tool_call(tool, {"command": "cat ~/somefile.txt"})
    assert decision.allowed is False
    assert "outside workspace root" in decision.reason


def test_dotslash_relative_path_inside_workspace_allowed(guard: ToolSecurityGuard, workspace: Path):
    (workspace / "sub").mkdir(exist_ok=True)
    tool = _FakeCommandTool()
    decision = guard.validate_tool_call(tool, {"command": "cat ./sub/file.txt"})
    assert decision.allowed is True


def test_malformed_command_invalid_command_token_denied(guard: ToolSecurityGuard):
    tool = _FakeCommandTool()
    decision = guard.validate_tool_call(tool, {"command": "echo 'unbalanced"})
    assert decision.allowed is False
