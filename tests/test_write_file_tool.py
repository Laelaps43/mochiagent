from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from agent.common.tools._utils import reset_workspace_root, set_workspace_root
from agent.common.tools.write_file_tool import WriteFileTool


def _result_map(result: object) -> dict[str, object]:
    assert isinstance(result, dict)
    return cast(dict[str, object], result)


@pytest.fixture
def workspace_guard() -> Iterator[None]:
    reset_workspace_root()
    yield
    reset_workspace_root()


def test_write_file_tool_metadata() -> None:
    tool = WriteFileTool()

    definition = tool.to_definition()

    assert tool.name == "write_file"
    assert tool.description == "Write content to a file."
    assert definition.name == "write_file"
    assert definition.required == ["path", "content"]


async def test_write_file_creates_parent_directories(tmp_path: Path) -> None:
    file_path = tmp_path / "nested" / "deep" / "hello.txt"
    content = "hé😊"

    result = _result_map(await WriteFileTool().execute(path=str(file_path), content=content))

    assert result["success"] is True
    assert result["path"] == str(file_path)
    assert result["append"] is False
    assert result["bytes_written"] == len(content.encode("utf-8", errors="ignore"))
    assert file_path.read_text(encoding="utf-8") == content


async def test_write_file_appends_when_requested(tmp_path: Path) -> None:
    file_path = tmp_path / "append.txt"
    _ = file_path.write_text("first", encoding="utf-8")

    result = _result_map(
        await WriteFileTool().execute(path=str(file_path), content="-second", append=True)
    )

    assert result["success"] is True
    assert result["append"] is True
    assert file_path.read_text(encoding="utf-8") == "first-second"


async def test_write_file_supports_custom_encoding(tmp_path: Path) -> None:
    file_path = tmp_path / "utf16.txt"
    content = "你好"

    result = _result_map(
        await WriteFileTool().execute(path=str(file_path), content=content, encoding="utf-16")
    )

    assert result["success"] is True
    assert result["bytes_written"] == len(content.encode("utf-16", errors="ignore"))
    assert file_path.read_text(encoding="utf-16") == content


async def test_write_file_rejects_workspace_violation(
    tmp_path: Path, workspace_guard: None
) -> None:
    _ = workspace_guard
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    blocked_path = tmp_path / "outside.txt"
    set_workspace_root(workspace)

    result = _result_map(await WriteFileTool().execute(path=str(blocked_path), content="blocked"))

    assert result["success"] is False
    assert "WORKSPACE_VIOLATION:" in cast(str, result["error"])


async def test_write_file_raises_for_directory_path(tmp_path: Path) -> None:
    directory = tmp_path / "folder"
    directory.mkdir()

    with pytest.raises(IsADirectoryError):
        _ = await WriteFileTool().execute(path=str(directory), content="x")
