from __future__ import annotations

import asyncio
from pathlib import Path
from typing import override

from agent.core.tools import Tool
from agent.common.tools._utils import validate_path_within_workspace


class EditFileTool(Tool):
    """
    编辑文件工具

    支持以下操作：
    1. 替换指定字符串（支持替换所有）
    2. 覆盖写入文件全部内容
    """

    @property
    @override
    def name(self) -> str:
        return "edit_file"

    @property
    @override
    def description(self) -> str:
        return "Edit file by replacing text or rewriting full content."

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
                "old_string": {"type": "string", "description": "Old text to replace"},
                "new_string": {"type": "string", "description": "Replacement text"},
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all matches",
                    "default": False,
                },
                "content": {"type": "string", "description": "Rewrite file with this full content"},
                "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
            },
            "required": ["path"],
        }

    @override
    async def execute(
        self,
        path: str = "",
        old_string: str | None = None,
        new_string: str | None = None,
        replace_all: bool = False,
        content: str | None = None,
        encoding: str = "utf-8",
        **kwargs: object,
    ) -> object:
        path_error = validate_path_within_workspace(path)
        if path_error:
            return {"success": False, "error": f"WORKSPACE_VIOLATION: {path_error}"}

        file_path = Path(path)
        if not file_path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        if file_path.is_dir():
            return {"success": False, "error": f"Path is a directory: {path}"}

        def _edit() -> dict[str, object]:
            original = file_path.read_text(encoding=encoding)

            if content is not None:
                updated = content
                count = 1
            else:
                if old_string is None or new_string is None:
                    return {
                        "success": False,
                        "error": "Either provide content, or provide old_string and new_string.",
                    }

                if replace_all:
                    count = original.count(old_string)
                    updated = original.replace(old_string, new_string)
                else:
                    count = 1 if old_string in original else 0
                    updated = original.replace(old_string, new_string, 1)

                if count == 0:
                    return {"success": False, "error": "old_string not found in file"}

            _ = file_path.write_text(updated, encoding=encoding)
            return {
                "success": True,
                "path": str(file_path),
                "replacements": count,
            }

        return await asyncio.to_thread(_edit)
