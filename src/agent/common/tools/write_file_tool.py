from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from agent.core.tools import Tool


class WriteFileTool(Tool):
    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file."

    @property
    def parameters_schema(self) -> Dict[str, Any]:
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

    async def execute(
        self,
        path: str,
        content: str,
        append: bool = False,
        encoding: str = "utf-8",
    ) -> Any:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with file_path.open(mode, encoding=encoding) as f:
            f.write(content)
        return {
            "success": True,
            "path": str(file_path),
            "bytes_written": len(content.encode(encoding, errors="ignore")),
            "append": append,
        }
