from __future__ import annotations

import asyncio
from typing import override

from agent.core.tools import Tool

from agent.common.tools.results import ExecResult

# Safety cap for subprocess output (1 MB).
_SAFETY_MAX_CHARS = 1024 * 1024


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
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            proc.kill()
            _ = await proc.wait()
            return ExecResult(
                success=False,
                exit_code=-1,
                output="Command timed out after 120s",
                truncated=False,
            )
        except asyncio.CancelledError:
            proc.kill()
            _ = await proc.wait()
            raise

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        combined = stdout + ("\n" if stdout and stderr else "") + stderr

        truncated = len(combined) > _SAFETY_MAX_CHARS
        if truncated:
            combined = combined[:_SAFETY_MAX_CHARS]

        return ExecResult(
            success=proc.returncode == 0,
            exit_code=proc.returncode,
            output=combined,
            truncated=truncated,
        )
