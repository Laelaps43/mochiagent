from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from agent.common.tools._utils import reset_workspace_root, set_workspace_root
from agent.common.tools.read_file_tool import ReadFileTool
from agent.common.tools.results import ReadFileSuccess, ToolError


@pytest.fixture
def workspace_guard() -> Iterator[None]:
    reset_workspace_root()
    yield
    reset_workspace_root()


def test_read_file_tool_metadata() -> None:
    tool = ReadFileTool()

    definition = tool.to_definition()

    assert tool.name == "read_file"
    assert tool.description == "Read file content from disk."
    assert definition.name == "read_file"
    assert definition.required == ["path"]


async def test_read_file_reads_requested_chunk(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    _ = file_path.write_text("abcdef", encoding="utf-8")

    result = await ReadFileTool().execute(path=str(file_path), offset=2, limit=3)

    assert isinstance(result, ReadFileSuccess)
    assert result.path == str(file_path)
    assert result.content == "cde"
    assert result.size_bytes == 6
    assert result.offset == 2
    assert result.limit == 3
    assert result.next_offset == 5
    assert result.eof is False
    assert result.truncated is True


async def test_read_file_uses_max_chars_when_limit_missing(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    _ = file_path.write_text("abcdef", encoding="utf-8")

    result = await ReadFileTool().execute(path=str(file_path), offset=-5, max_chars=2)

    assert isinstance(result, ReadFileSuccess)
    assert result.content == "ab"
    assert result.offset == 0
    assert result.limit == 2
    assert result.next_offset == 2
    assert result.eof is False
    assert result.truncated is True


async def test_read_file_clamps_offset_and_limit(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    _ = file_path.write_text("abc", encoding="utf-8")

    result = await ReadFileTool().execute(path=str(file_path), offset=99, limit=0)

    assert isinstance(result, ReadFileSuccess)
    assert result.content == ""
    assert result.offset == 3
    assert result.limit == 1
    assert result.next_offset == 3
    assert result.eof is True
    assert result.truncated is False


async def test_read_file_rejects_missing_path(tmp_path: Path) -> None:
    file_path = tmp_path / "missing.txt"

    result = await ReadFileTool().execute(path=str(file_path))

    assert isinstance(result, ToolError)
    assert result.success is False
    assert f"File not found: {file_path}" in result.error


async def test_read_file_rejects_directory_path(tmp_path: Path) -> None:
    result = await ReadFileTool().execute(path=str(tmp_path))

    assert isinstance(result, ToolError)
    assert f"Path is a directory: {tmp_path}" in result.error


async def test_read_file_rejects_workspace_violation(tmp_path: Path, workspace_guard: None) -> None:
    _ = workspace_guard
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_file = tmp_path / "outside.txt"
    _ = outside_file.write_text("secret", encoding="utf-8")
    set_workspace_root(workspace)

    result = await ReadFileTool().execute(path=str(outside_file))

    assert isinstance(result, ToolError)
    assert result.success is False
    assert "WORKSPACE_VIOLATION:" in result.error
    assert "outside the workspace root" in result.error
