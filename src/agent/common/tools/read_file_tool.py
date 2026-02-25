from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from agent.core.tools import Tool


class ReadFileTool(Tool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read file content from disk."

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
                "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
                "max_chars": {
                    "type": "integer",
                    "description": "Max chars to return",
                    "default": 100000,
                },
                "offset": {"type": "integer", "description": "Character offset", "default": 0},
                "limit": {"type": "integer", "description": "Character limit for chunked read"},
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        encoding: str = "utf-8",
        max_chars: int = 100000,
        offset: int = 0,
        limit: int | None = None,
    ) -> Any:
        file_path = Path(path)
        if not file_path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        if file_path.is_dir():
            return {"success": False, "error": f"Path is a directory: {path}"}

        content = file_path.read_text(encoding=encoding)
        total_size = len(content)
        safe_offset = max(0, offset)
        if safe_offset > total_size:
            safe_offset = total_size

        if limit is not None:
            safe_limit = max(1, limit)
        else:
            safe_limit = max(1, max_chars)

        chunk = content[safe_offset : safe_offset + safe_limit]
        next_offset = safe_offset + len(chunk)
        eof = next_offset >= total_size
        truncated = not eof
        return {
            "success": True,
            "path": str(file_path),
            "content": chunk,
            "truncated": truncated,
            "size": total_size,
            "offset": safe_offset,
            "limit": safe_limit,
            "next_offset": next_offset,
            "eof": eof,
        }
