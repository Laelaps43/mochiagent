from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from agent.core.tools import Tool


class EditFileTool(Tool):
    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "Edit file by replacing text or rewriting full content."

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

    async def execute(
        self,
        path: str,
        old_string: str | None = None,
        new_string: str | None = None,
        replace_all: bool = False,
        content: str | None = None,
        encoding: str = "utf-8",
    ) -> Any:
        file_path = Path(path)
        if not file_path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        if file_path.is_dir():
            return {"success": False, "error": f"Path is a directory: {path}"}

        original = file_path.read_text(encoding=encoding)

        if content is not None:
            updated = content
            replacements = 1
        else:
            if old_string is None or new_string is None:
                return {
                    "success": False,
                    "error": "Either provide content, or provide old_string and new_string.",
                }

            if replace_all:
                replacements = original.count(old_string)
                updated = original.replace(old_string, new_string)
            else:
                replacements = 1 if old_string in original else 0
                updated = original.replace(old_string, new_string, 1)

            if replacements == 0:
                return {"success": False, "error": "old_string not found in file"}

        file_path.write_text(updated, encoding=encoding)
        return {
            "success": True,
            "path": str(file_path),
            "replacements": replacements,
        }
