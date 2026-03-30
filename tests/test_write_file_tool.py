from __future__ import annotations

from pathlib import Path

from agent.common.tools.write_file_tool import WriteFileTool
from agent.common.tools.results import ToolError, WriteFileSuccess


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

    result = await WriteFileTool().execute(path=str(file_path), content=content)

    assert isinstance(result, WriteFileSuccess)
    assert result.path == str(file_path)
    assert result.append is False
    assert result.bytes_written == len(content.encode("utf-8", errors="ignore"))
    assert file_path.read_text(encoding="utf-8") == content


async def test_write_file_appends_when_requested(tmp_path: Path) -> None:
    file_path = tmp_path / "append.txt"
    _ = file_path.write_text("first", encoding="utf-8")

    result = await WriteFileTool().execute(path=str(file_path), content="-second", append=True)

    assert isinstance(result, WriteFileSuccess)
    assert result.append is True
    assert file_path.read_text(encoding="utf-8") == "first-second"


async def test_write_file_supports_custom_encoding(tmp_path: Path) -> None:
    file_path = tmp_path / "utf16.txt"
    content = "你好"

    result = await WriteFileTool().execute(path=str(file_path), content=content, encoding="utf-16")

    assert isinstance(result, WriteFileSuccess)
    assert result.bytes_written == len(content.encode("utf-16", errors="ignore"))
    assert file_path.read_text(encoding="utf-16") == content


async def test_write_file_returns_error_for_directory_path(tmp_path: Path) -> None:
    directory = tmp_path / "folder"
    directory.mkdir()

    result = await WriteFileTool().execute(path=str(directory), content="x")

    assert isinstance(result, ToolError)
    assert "directory" in result.error.lower()
