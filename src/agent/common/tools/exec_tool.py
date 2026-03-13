from __future__ import annotations

import asyncio
from typing import override

from agent.core.tools import Tool

from ._utils import truncate_text, validate_path_within_workspace
from .results import ExecResult, ToolError


class ExecTool(Tool):
    def __init__(self, max_output_chars: int = 20000):
        self.max_output_chars: int = max_output_chars

    @property
    @override
    def name(self) -> str:
        return "exec"

    @property
    @override
    def description(self) -> str:
        return "Execute shell command."

    @property
    @override
    def parameters_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command",
                    "x-shell-command": True,
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory",
                    "x-workspace-cwd": True,
                },
            },
            "required": ["command"],
        }

    @override
    async def execute(
        self, command: str = "", workdir: str | None = None, **kwargs: object
    ) -> object:
        if workdir:
            path_error = validate_path_within_workspace(workdir)
            if path_error:
                return ToolError(error=f"WORKSPACE_VIOLATION: {path_error}")
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await proc.communicate()
        except asyncio.CancelledError:
            # 外层 wait_for 超时会取消此协程，确保子进程被终止
            proc.kill()
            _ = await proc.wait()
            raise
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        combined = stdout + ("\n" if stdout and stderr else "") + stderr
        truncated_output, truncated = truncate_text(combined, self.max_output_chars)

        return ExecResult(
            success=proc.returncode == 0,
            exit_code=proc.returncode,
            output=truncated_output,
            truncated=truncated,
        )
