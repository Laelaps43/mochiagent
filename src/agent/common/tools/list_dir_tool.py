from __future__ import annotations

from pathlib import Path
from typing import override

from agent.core.tools import Tool
from agent.common.tools._utils import validate_path_within_workspace


class ListDirTool(Tool):
    @property
    @override
    def name(self) -> str:
        return "list_dir"

    @property
    @override
    def description(self) -> str:
        return "List directory entries."

    @property
    @override
    def parameters_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path",
                    "default": ".",
                    "x-workspace-path": True,
                },
                "max_entries": {
                    "type": "integer",
                    "description": "Max entries to return",
                    "default": 200,
                },
            },
            "required": [],
        }

    @override
    async def execute(self, path: str = ".", max_entries: int = 200, **kwargs: object) -> object:
        path_error = validate_path_within_workspace(path)
        if path_error:
            return {"success": False, "error": f"WORKSPACE_VIOLATION: {path_error}"}

        dir_path = Path(path)
        if not dir_path.exists():
            return {"success": False, "error": f"Directory not found: {path}"}
        if not dir_path.is_dir():
            return {"success": False, "error": f"Path is not a directory: {path}"}

        entries: list[str] = []
        for entry in sorted(dir_path.iterdir(), key=lambda x: x.name):
            entries.append(entry.name + ("/" if entry.is_dir() else ""))
            if len(entries) >= max_entries:
                break

        return {
            "success": True,
            "path": str(dir_path),
            "entries": entries,
            "truncated": len(entries) >= max_entries,
        }
