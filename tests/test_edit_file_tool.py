from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from agent.common.tools._utils import reset_workspace_root, set_workspace_root
from agent.common.tools.edit_file_tool import EditFileTool


def _result_map(result: object) -> dict[str, object]:
    assert isinstance(result, dict)
    return cast(dict[str, object], result)


@pytest.fixture
def workspace_guard() -> Iterator[None]:
    reset_workspace_root()
    yield
    reset_workspace_root()


def test_edit_file_tool_metadata() -> None:
    tool = EditFileTool()

    definition = tool.to_definition()

    assert tool.name == "edit_file"
    assert tool.description == "Edit file by replacing text or rewriting full content."
    assert definition.name == "edit_file"
    assert definition.required == ["path"]


async def test_edit_file_rewrites_entire_content(tmp_path: Path) -> None:
    file_path = tmp_path / "rewrite.txt"
    _ = file_path.write_text("old", encoding="utf-8")

    result = _result_map(await EditFileTool().execute(path=str(file_path), content="new body"))

    assert result == {"success": True, "path": str(file_path), "replacements": 1}
    assert file_path.read_text(encoding="utf-8") == "new body"


async def test_edit_file_replaces_first_match_only(tmp_path: Path) -> None:
    file_path = tmp_path / "replace_once.txt"
    _ = file_path.write_text("one two one", encoding="utf-8")

    result = _result_map(
        await EditFileTool().execute(path=str(file_path), old_string="one", new_string="1")
    )

    assert result == {"success": True, "path": str(file_path), "replacements": 1}
    assert file_path.read_text(encoding="utf-8") == "1 two one"


async def test_edit_file_replaces_all_matches(tmp_path: Path) -> None:
    file_path = tmp_path / "replace_all.txt"
    _ = file_path.write_text("red red blue red", encoding="utf-8")

    result = _result_map(
        await EditFileTool().execute(
            path=str(file_path),
            old_string="red",
            new_string="green",
            replace_all=True,
        )
    )

    assert result == {"success": True, "path": str(file_path), "replacements": 3}
    assert file_path.read_text(encoding="utf-8") == "green green blue green"


async def test_edit_file_requires_content_or_replacement_pair(tmp_path: Path) -> None:
    file_path = tmp_path / "missing_args.txt"
    _ = file_path.write_text("text", encoding="utf-8")

    result = _result_map(await EditFileTool().execute(path=str(file_path), old_string="text"))

    assert result == {
        "success": False,
        "error": "Either provide content, or provide old_string and new_string.",
    }


async def test_edit_file_reports_missing_old_string(tmp_path: Path) -> None:
    file_path = tmp_path / "missing_old.txt"
    _ = file_path.write_text("hello", encoding="utf-8")

    result = _result_map(
        await EditFileTool().execute(path=str(file_path), old_string="bye", new_string="ciao")
    )

    assert result == {"success": False, "error": "old_string not found in file"}


async def test_edit_file_rejects_missing_path(tmp_path: Path) -> None:
    file_path = tmp_path / "missing.txt"

    result = _result_map(
        await EditFileTool().execute(path=str(file_path), old_string="a", new_string="b")
    )

    assert result == {"success": False, "error": f"File not found: {file_path}"}


async def test_edit_file_rejects_directory_path(tmp_path: Path) -> None:
    result = _result_map(
        await EditFileTool().execute(path=str(tmp_path), old_string="a", new_string="b")
    )

    assert result == {"success": False, "error": f"Path is a directory: {tmp_path}"}


async def test_edit_file_rejects_workspace_violation(tmp_path: Path, workspace_guard: None) -> None:
    _ = workspace_guard
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_file = tmp_path / "outside.txt"
    _ = outside_file.write_text("secret", encoding="utf-8")
    set_workspace_root(workspace)

    result = _result_map(
        await EditFileTool().execute(path=str(outside_file), old_string="secret", new_string="x")
    )

    assert result["success"] is False
    assert "WORKSPACE_VIOLATION:" in cast(str, result["error"])
