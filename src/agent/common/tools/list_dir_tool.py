from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from agent.core.tools import Tool


class ListDirTool(Tool):
    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List directory entries."

    @property
    def parameters_schema(self) -> Dict[str, Any]:
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

    async def execute(self, path: str = ".", max_entries: int = 200) -> Any:
        dir_path = Path(path)
        if not dir_path.exists():
            return {"success": False, "error": f"Directory not found: {path}"}
        if not dir_path.is_dir():
            return {"success": False, "error": f"Path is not a directory: {path}"}

        entries = []
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
