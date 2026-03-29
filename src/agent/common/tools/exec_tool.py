from __future__ import annotations

import asyncio
from typing import override

from agent.core.tools import Tool

from ._utils import validate_path_within_workspace
from .results import ExecResult, ToolError

# 安全上限：防止子进程输出无限大导致内存爆炸
_SAFETY_MAX_CHARS = 1024 * 1024  # 1MB


class ExecTool(Tool):
    @property
    @override
    def timeout(self) -> int | None:
        return 120  # 2 minutes

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
            proc.kill()
            _ = await proc.wait()
            raise
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        combined = stdout + ("\n" if stdout and stderr else "") + stderr

        # 安全上限截断（防止内存爆炸），postprocessor 会做 artifact 保存
        truncated = len(combined) > _SAFETY_MAX_CHARS
        if truncated:
            combined = combined[:_SAFETY_MAX_CHARS]

        return ExecResult(
            success=proc.returncode == 0,
            exit_code=proc.returncode,
            output=combined,
            truncated=truncated,
        )
