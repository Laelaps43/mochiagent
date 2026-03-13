from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from agent.common.tools._utils import reset_workspace_root, set_workspace_root
from agent.common.tools.list_dir_tool import ListDirTool
from agent.common.tools.results import ListDirSuccess, ToolError


@pytest.fixture
def workspace_guard() -> Iterator[None]:
    reset_workspace_root()
    yield
    reset_workspace_root()


def test_list_dir_tool_metadata() -> None:
    tool = ListDirTool()

    definition = tool.to_definition()

    assert tool.name == "list_dir"
    assert tool.description == "List directory entries."
    assert definition.name == "list_dir"
    assert definition.required == []


async def test_list_dir_returns_sorted_entries_with_directory_suffix(tmp_path: Path) -> None:
    alpha_dir = tmp_path / "alpha"
    alpha_dir.mkdir()
    _ = (tmp_path / "beta.txt").write_text("b", encoding="utf-8")
    _ = (tmp_path / "aardvark.txt").write_text("a", encoding="utf-8")

    result = await ListDirTool().execute(path=str(tmp_path), max_entries=10)

    assert isinstance(result, ListDirSuccess)
    assert result.path == str(tmp_path)
    assert result.entries == ["aardvark.txt", "alpha/", "beta.txt"]
    assert result.truncated is False


async def test_list_dir_truncates_when_entry_limit_reached(tmp_path: Path) -> None:
    _ = (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    _ = (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    _ = (tmp_path / "c.txt").write_text("c", encoding="utf-8")

    result = await ListDirTool().execute(path=str(tmp_path), max_entries=2)

    assert isinstance(result, ListDirSuccess)
    assert result.entries == ["a.txt", "b.txt"]
    assert result.truncated is True


async def test_list_dir_rejects_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    result = await ListDirTool().execute(path=str(missing))

    assert isinstance(result, ToolError)
    assert f"Directory not found: {missing}" in result.error


async def test_list_dir_rejects_file_path(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    _ = file_path.write_text("content", encoding="utf-8")

    result = await ListDirTool().execute(path=str(file_path))

    assert isinstance(result, ToolError)
    assert f"Path is not a directory: {file_path}" in result.error


async def test_list_dir_rejects_workspace_violation(tmp_path: Path, workspace_guard: None) -> None:
    _ = workspace_guard
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    set_workspace_root(workspace)

    result = await ListDirTool().execute(path=str(outside_dir))

    assert isinstance(result, ToolError)
    assert result.success is False
    assert "WORKSPACE_VIOLATION:" in result.error
