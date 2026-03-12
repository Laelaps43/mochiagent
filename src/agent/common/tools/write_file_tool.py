from __future__ import annotations

import asyncio
from pathlib import Path
from typing import override

from agent.core.tools import Tool
from agent.common.tools._utils import validate_path_within_workspace


class WriteFileTool(Tool):
    @property
    @override
    def name(self) -> str:
        return "write_file"

    @property
    @override
    def description(self) -> str:
        return "Write content to a file."

    @property
    @override
    def parameters_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path",
                    "x-workspace-path": True,
                },
                "content": {"type": "string", "description": "Content to write"},
                "append": {"type": "boolean", "description": "Append mode", "default": False},
                "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
            },
            "required": ["path", "content"],
        }

    @override
    async def execute(
        self,
        path: str = "",
        content: str = "",
        append: bool = False,
        encoding: str = "utf-8",
        **kwargs: object,
    ) -> object:
        path_error = validate_path_within_workspace(path)
        if path_error:
            return {"success": False, "error": f"WORKSPACE_VIOLATION: {path_error}"}

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"

        def _write() -> None:
            with file_path.open(mode, encoding=encoding) as f:
                _ = f.write(content)

        await asyncio.to_thread(_write)
        return {
            "success": True,
            "path": str(file_path),
            "bytes_written": len(content.encode(encoding, errors="ignore")),
            "append": append,
        }
