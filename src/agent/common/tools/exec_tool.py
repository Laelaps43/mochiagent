from __future__ import annotations

import asyncio
from typing import Any, Dict

from agent.core.tools import Tool

from ._utils import truncate_text


class ExecTool(Tool):
    def __init__(self, max_output_chars: int = 20000):
        self.max_output_chars = max_output_chars

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Execute shell command."

    @property
    def parameters_schema(self) -> Dict[str, Any]:
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

    async def execute(self, command: str, workdir: str | None = None) -> Any:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        combined = stdout + ("\n" if stdout and stderr else "") + stderr
        truncated_output, truncated = truncate_text(combined, self.max_output_chars)

        return {
            "success": proc.returncode == 0,
            "exit_code": proc.returncode,
            "output": truncated_output,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": truncated,
        }
