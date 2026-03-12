from __future__ import annotations

from pathlib import Path
from typing import override

from agent.core.tools import Tool


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
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with file_path.open(mode, encoding=encoding) as f:
            _ = f.write(content)
        return {
            "success": True,
            "path": str(file_path),
            "bytes_written": len(content.encode(encoding, errors="ignore")),
            "append": append,
        }
