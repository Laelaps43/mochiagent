from __future__ import annotations

import asyncio
from pathlib import Path
from typing import override

from agent.core.tools import Tool
from agent.common.tools._utils import validate_path_within_workspace
from agent.common.tools.results import ReadFileSuccess, ToolError


class ReadFileTool(Tool):
    """
    读取文件内容

    支持按需读取文件内容，避免大文件全量加载
    """

    @property
    @override
    def name(self) -> str:
        return "read_file"

    @property
    @override
    def description(self) -> str:
        return "Read file content from disk."

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

    @override
    async def execute(
        self,
        path: str = "",
        encoding: str = "utf-8",
        max_chars: int = 100000,
        offset: int = 0,
        limit: int | None = None,
        **kwargs: object,
    ) -> object:
        path_error = validate_path_within_workspace(path)
        if path_error:
            return ToolError(error=f"WORKSPACE_VIOLATION: {path_error}")

        file_path = Path(path)
        if not file_path.exists():
            return ToolError(error=f"File not found: {path}")
        if file_path.is_dir():
            return ToolError(error=f"Path is a directory: {path}")

        total_bytes = file_path.stat().st_size
        safe_offset = max(0, offset)

        if limit is not None:
            safe_limit = max(1, limit)
        else:
            safe_limit = max(1, max_chars)

        def _read() -> tuple[str, bool, int]:
            with file_path.open("r", encoding=encoding) as f:
                if safe_offset > 0:
                    skipped = f.read(safe_offset)
                    # If we overshot (offset > total chars), clamp
                    if len(skipped) < safe_offset:
                        return "", True, len(skipped)
                data = f.read(safe_limit)
                is_eof = f.read(1) == ""
            actual_offset = safe_offset
            return data, is_eof, actual_offset

        chunk, eof, actual_offset = await asyncio.to_thread(_read)

        next_offset = actual_offset + len(chunk)
        truncated = not eof
        return ReadFileSuccess(
            path=str(file_path),
            content=chunk,
            truncated=truncated,
            size_bytes=total_bytes,
            offset=actual_offset,
            limit=safe_limit,
            next_offset=next_offset,
            eof=eof,
        )
