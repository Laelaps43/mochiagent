from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from agent.common.tools.edit_file_tool import EditFileTool
from agent.common.tools.exec_tool import ExecTool
from agent.common.tools.list_dir_tool import ListDirTool
from agent.common.tools.read_file_tool import ReadFileTool
from agent.common.tools.web_fetch_tool import WebFetchTool
from agent.common.tools.write_file_tool import WriteFileTool


@pytest.mark.asyncio
async def test_read_file_tool_schema_and_behavior(tmp_path: Path):
    tool = ReadFileTool()
    schema = tool.parameters_schema
    assert schema["properties"]["path"]["x-workspace-path"] is True

    target = tmp_path / "a.txt"
    target.write_text("hello world", encoding="utf-8")

    ok = await tool.execute(path=str(target), max_chars=5)
    assert ok["success"] is True
    assert ok["content"] == "hello"
    assert ok["truncated"] is True

    paged = await tool.execute(path=str(target), offset=6, limit=5)
    assert paged["success"] is True
    assert paged["content"] == "world"
    assert paged["eof"] is True


@pytest.mark.asyncio
async def test_write_file_tool_schema_and_behavior(tmp_path: Path):
    tool = WriteFileTool()
    schema = tool.parameters_schema
    assert schema["properties"]["path"]["x-workspace-path"] is True

    target = tmp_path / "b.txt"
    first = await tool.execute(path=str(target), content="abc")
    assert first["success"] is True
    assert target.read_text(encoding="utf-8") == "abc"

    second = await tool.execute(path=str(target), content="def", append=True)
    assert second["success"] is True
    assert target.read_text(encoding="utf-8") == "abcdef"


@pytest.mark.asyncio
async def test_edit_file_tool_schema_and_behavior(tmp_path: Path):
    tool = EditFileTool()
    schema = tool.parameters_schema
    assert schema["properties"]["path"]["x-workspace-path"] is True

    target = tmp_path / "c.txt"
    target.write_text("foo bar foo", encoding="utf-8")

    ok = await tool.execute(path=str(target), old_string="foo", new_string="baz", replace_all=False)
    assert ok["success"] is True
    assert ok["replacements"] == 1
    assert target.read_text(encoding="utf-8") == "baz bar foo"

    fail = await tool.execute(path=str(target), old_string="not-found", new_string="x")
    assert fail["success"] is False
    assert "old_string not found" in fail["error"]


@pytest.mark.asyncio
async def test_list_dir_tool_schema_and_behavior(tmp_path: Path):
    tool = ListDirTool()
    schema = tool.parameters_schema
    assert schema["properties"]["path"]["x-workspace-path"] is True

    (tmp_path / "file1.txt").write_text("1", encoding="utf-8")
    (tmp_path / "dir1").mkdir()

    ok = await tool.execute(path=str(tmp_path), max_entries=10)
    assert ok["success"] is True
    assert "file1.txt" in ok["entries"]
    assert "dir1/" in ok["entries"]


@pytest.mark.asyncio
async def test_exec_tool_schema_and_behavior():
    tool = ExecTool(max_output_chars=5)
    schema = tool.parameters_schema
    assert schema["properties"]["workdir"]["x-workspace-cwd"] is True
    assert schema["properties"]["command"]["x-shell-command"] is True

    ok = await tool.execute(command="printf 'abcdef'")
    assert ok["output"] == "abcde"
    assert ok["truncated"] is True


class _FakeResponse:
    def __init__(self):
        self.status_code = 200
        self.text = "hello web fetch"
        self.headers = {"content-type": "text/plain"}
        self.url = "https://example.com/test"


@pytest.mark.asyncio
async def test_web_fetch_tool_behavior(monkeypatch: pytest.MonkeyPatch):
    tool = WebFetchTool(max_chars=5)

    async def fake_get(self, url, *args, **kwargs):
        return _FakeResponse()

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    bad = await tool.execute(url="ftp://example.com")
    assert bad["success"] is False

    ok = await tool.execute(url="https://example.com")
    assert ok["success"] is True
    assert ok["content"] == "hello"
    assert ok["truncated"] is True
