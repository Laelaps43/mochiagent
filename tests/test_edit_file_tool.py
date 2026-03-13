from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from agent.common.tools._utils import reset_workspace_root, set_workspace_root
from agent.common.tools.edit_file_tool import EditFileTool
from agent.common.tools.results import EditFileSuccess, ToolError


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

    result = await EditFileTool().execute(path=str(file_path), content="new body")

    assert isinstance(result, EditFileSuccess)
    assert result.path == str(file_path)
    assert result.replacements == 1
    assert result.warning is None
    assert file_path.read_text(encoding="utf-8") == "new body"


async def test_edit_file_replaces_first_match_only(tmp_path: Path) -> None:
    file_path = tmp_path / "replace_once.txt"
    _ = file_path.write_text("one two one", encoding="utf-8")

    result = await EditFileTool().execute(path=str(file_path), old_string="one", new_string="1")

    assert isinstance(result, EditFileSuccess)
    assert result.replacements == 1
    assert result.warning is not None  # old_string matched 2 times but only first replaced
    assert file_path.read_text(encoding="utf-8") == "1 two one"


async def test_edit_file_replaces_all_matches(tmp_path: Path) -> None:
    file_path = tmp_path / "replace_all.txt"
    _ = file_path.write_text("red red blue red", encoding="utf-8")

    result = await EditFileTool().execute(
        path=str(file_path),
        old_string="red",
        new_string="green",
        replace_all=True,
    )

    assert isinstance(result, EditFileSuccess)
    assert result.path == str(file_path)
    assert result.replacements == 3
    assert result.warning is None
    assert file_path.read_text(encoding="utf-8") == "green green blue green"


async def test_edit_file_requires_content_or_replacement_pair(tmp_path: Path) -> None:
    file_path = tmp_path / "missing_args.txt"
    _ = file_path.write_text("text", encoding="utf-8")

    result = await EditFileTool().execute(path=str(file_path), old_string="text")

    assert isinstance(result, ToolError)
    assert "Either provide content, or provide old_string and new_string." in result.error


async def test_edit_file_reports_missing_old_string(tmp_path: Path) -> None:
    file_path = tmp_path / "missing_old.txt"
    _ = file_path.write_text("hello", encoding="utf-8")

    result = await EditFileTool().execute(path=str(file_path), old_string="bye", new_string="ciao")

    assert isinstance(result, ToolError)
    assert "old_string not found in file" in result.error


async def test_edit_file_rejects_missing_path(tmp_path: Path) -> None:
    file_path = tmp_path / "missing.txt"

    result = await EditFileTool().execute(path=str(file_path), old_string="a", new_string="b")

    assert isinstance(result, ToolError)
    assert f"File not found: {file_path}" in result.error


async def test_edit_file_rejects_directory_path(tmp_path: Path) -> None:
    result = await EditFileTool().execute(path=str(tmp_path), old_string="a", new_string="b")

    assert isinstance(result, ToolError)
    assert f"Path is a directory: {tmp_path}" in result.error


async def test_edit_file_rejects_workspace_violation(tmp_path: Path, workspace_guard: None) -> None:
    _ = workspace_guard
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_file = tmp_path / "outside.txt"
    _ = outside_file.write_text("secret", encoding="utf-8")
    set_workspace_root(workspace)

    result = await EditFileTool().execute(
        path=str(outside_file), old_string="secret", new_string="x"
    )

    assert isinstance(result, ToolError)
    assert result.success is False
    assert "WORKSPACE_VIOLATION:" in result.error
