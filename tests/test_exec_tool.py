from __future__ import annotations

import asyncio
from pathlib import Path
from shlex import quote
import sys
from typing import cast, final

import pytest
from pytest import MonkeyPatch

from agent.common.tools.exec_tool import ExecTool


def _result_map(result: object) -> dict[str, object]:
    assert isinstance(result, dict)
    return cast(dict[str, object], result)


def test_exec_tool_metadata() -> None:
    tool = ExecTool(max_output_chars=123)

    definition = tool.to_definition()

    assert tool.name == "exec"
    assert tool.description == "Execute shell command."
    assert tool.max_output_chars == 123
    assert definition.name == "exec"
    assert definition.required == ["command"]


async def test_exec_tool_runs_successful_command(tmp_path: Path) -> None:
    command = f"{quote(sys.executable)} -c \"print('hello from exec')\""

    result = _result_map(await ExecTool().execute(command=command, workdir=str(tmp_path)))

    assert result["success"] is True
    assert result["exit_code"] == 0
    assert result["stdout"] == "hello from exec\n"
    assert result["stderr"] == ""
    assert result["output"] == "hello from exec\n"
    assert result["truncated"] is False


async def test_exec_tool_reports_stderr_and_nonzero_exit() -> None:
    command = (
        f'{quote(sys.executable)} -c "import sys; '
        "print('out'); print('err', file=sys.stderr); raise SystemExit(3)\""
    )

    result = _result_map(await ExecTool().execute(command=command))

    assert result["success"] is False
    assert result["exit_code"] == 3
    assert result["stdout"] == "out\n"
    assert result["stderr"] == "err\n"
    assert result["output"] == "out\n\nerr\n"
    assert result["truncated"] is False


async def test_exec_tool_uses_requested_workdir(tmp_path: Path) -> None:
    command = f'{quote(sys.executable)} -c "from pathlib import Path; print(Path.cwd())"'

    result = _result_map(await ExecTool().execute(command=command, workdir=str(tmp_path)))

    assert result["success"] is True
    assert cast(str, result["stdout"]).strip() == str(tmp_path)


async def test_exec_tool_truncates_combined_output() -> None:
    command = f"{quote(sys.executable)} -c \"print('abcdefghij', end='')\""

    result = _result_map(await ExecTool(max_output_chars=5).execute(command=command))

    assert result["success"] is True
    assert result["stdout"] == "abcdefghij"
    assert result["output"] == "abcde"
    assert result["truncated"] is True


async def test_exec_tool_kills_process_when_cancelled(monkeypatch: MonkeyPatch) -> None:
    @final
    class _FakeProcess:
        def __init__(self) -> None:
            self.returncode: int = -9
            self.killed: bool = False
            self.waited: bool = False

        def kill(self) -> None:
            self.killed = True

        async def communicate(self) -> tuple[bytes, bytes]:
            raise asyncio.CancelledError

        async def wait(self) -> int:
            self.waited = True
            return self.returncode

    fake_process = _FakeProcess()

    async def _fake_create_subprocess_shell(*_args: object, **_kwargs: object) -> _FakeProcess:
        return fake_process

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _fake_create_subprocess_shell)

    with pytest.raises(asyncio.CancelledError):
        _ = await ExecTool().execute(command="ignored")

    assert fake_process.killed is True
    assert fake_process.waited is True
