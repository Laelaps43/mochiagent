from __future__ import annotations

import asyncio
from pathlib import Path
from shlex import quote
import sys
from typing import final

import pytest
from pytest import MonkeyPatch

from agent.common.tools.exec_tool import ExecTool
from agent.common.tools.results import ExecResult


def test_exec_tool_metadata() -> None:
    tool = ExecTool()

    definition = tool.to_definition()

    assert tool.name == "exec"
    assert tool.description == "Execute shell command."
    assert tool.timeout == 120
    assert definition.name == "exec"
    assert definition.required == ["command"]


async def test_exec_tool_runs_successful_command(tmp_path: Path) -> None:
    command = f"{quote(sys.executable)} -c \"print('hello from exec')\""

    result = await ExecTool().execute(command=command, workdir=str(tmp_path))

    assert isinstance(result, ExecResult)
    assert result.success is True
    assert result.exit_code == 0
    assert result.output == "hello from exec\n"
    assert result.truncated is False


async def test_exec_tool_reports_stderr_and_nonzero_exit() -> None:
    command = (
        f'{quote(sys.executable)} -c "import sys; '
        "print('out'); print('err', file=sys.stderr); raise SystemExit(3)\""
    )

    result = await ExecTool().execute(command=command)

    assert isinstance(result, ExecResult)
    assert result.success is False
    assert result.exit_code == 3
    assert result.output == "out\n\nerr\n"
    assert result.truncated is False


async def test_exec_tool_uses_requested_workdir(tmp_path: Path) -> None:
    command = f'{quote(sys.executable)} -c "from pathlib import Path; print(Path.cwd())"'

    result = await ExecTool().execute(command=command, workdir=str(tmp_path))

    assert isinstance(result, ExecResult)
    assert result.success is True
    assert result.output.strip() == str(tmp_path)


async def test_exec_tool_does_not_truncate_small_output() -> None:
    command = f"{quote(sys.executable)} -c \"print('abcdefghij', end='')\""

    result = await ExecTool().execute(command=command)

    assert isinstance(result, ExecResult)
    assert result.success is True
    assert result.output == "abcdefghij"
    assert result.truncated is False


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
