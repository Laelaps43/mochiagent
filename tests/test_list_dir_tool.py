from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from agent.common.tools._utils import reset_workspace_root, set_workspace_root
from agent.common.tools.list_dir_tool import ListDirTool


def _result_map(result: object) -> dict[str, object]:
    assert isinstance(result, dict)
    return cast(dict[str, object], result)


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

    result = _result_map(await ListDirTool().execute(path=str(tmp_path), max_entries=10))
    entries = cast(list[str], result["entries"])

    assert result["success"] is True
    assert result["path"] == str(tmp_path)
    assert entries == ["aardvark.txt", "alpha/", "beta.txt"]
    assert result["truncated"] is False


async def test_list_dir_truncates_when_entry_limit_reached(tmp_path: Path) -> None:
    _ = (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    _ = (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    _ = (tmp_path / "c.txt").write_text("c", encoding="utf-8")

    result = _result_map(await ListDirTool().execute(path=str(tmp_path), max_entries=2))
    entries = cast(list[str], result["entries"])

    assert entries == ["a.txt", "b.txt"]
    assert result["truncated"] is True


async def test_list_dir_rejects_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    result = _result_map(await ListDirTool().execute(path=str(missing)))

    assert result == {"success": False, "error": f"Directory not found: {missing}"}


async def test_list_dir_rejects_file_path(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    _ = file_path.write_text("content", encoding="utf-8")

    result = _result_map(await ListDirTool().execute(path=str(file_path)))

    assert result == {"success": False, "error": f"Path is not a directory: {file_path}"}


async def test_list_dir_rejects_workspace_violation(tmp_path: Path, workspace_guard: None) -> None:
    _ = workspace_guard
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    set_workspace_root(workspace)

    result = _result_map(await ListDirTool().execute(path=str(outside_dir)))

    assert result["success"] is False
    assert "WORKSPACE_VIOLATION:" in cast(str, result["error"])
