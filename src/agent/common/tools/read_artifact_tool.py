"""Read artifact content via StorageProvider."""

from __future__ import annotations

from typing import Literal, override

from agent.core.storage.provider import StorageProvider
from agent.core.tools import Tool
from agent.common.tools.results import ToolError


class ReadArtifactTool(Tool):
    """Read artifact content stored by tool result post-processor.

    Uses StorageProvider to read artifacts, supporting any storage backend.
    """

    def __init__(self, storage: StorageProvider) -> None:
        self._storage: StorageProvider = storage

    @property
    @override
    def sandbox_mode(self) -> Literal["subprocess", "inprocess"]:
        return "inprocess"

    @property
    @override
    def name(self) -> str:
        return "read_artifact"

    @property
    @override
    def description(self) -> str:
        return (
            "Read artifact content that was saved when tool output was truncated. "
            "Use this to retrieve the complete output of a previous tool call. "
            "Supports offset and limit for chunked reading of large artifacts."
        )

    @property
    @override
    def parameters_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "artifact_ref": {
                    "type": "string",
                    "description": "Artifact reference (e.g. artifact://session_id/artifact_name)",
                },
                "offset": {
                    "type": "integer",
                    "description": "Character offset for chunked reading",
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max characters to return",
                    "default": 100000,
                },
            },
            "required": ["artifact_ref"],
        }

    @override
    async def execute(
        self,
        artifact_ref: str = "",
        offset: int = 0,
        limit: int = 100000,
        **kwargs: object,
    ) -> object:
        if not artifact_ref:
            return ToolError(error="artifact_ref is required")

        try:
            result = await self._storage.read_artifact(
                artifact_ref=artifact_ref,
                offset=offset,
                limit=limit,
            )
            if not result.success:
                return ToolError(error=result.error or "Failed to read artifact")
            return {
                "content": result.content,
                "offset": result.offset,
                "limit": result.limit,
                "next_offset": result.next_offset,
                "eof": result.eof,
                "size": result.size,
            }
        except NotImplementedError:
            return ToolError(error="Storage backend does not support artifact reading")
        except Exception as e:
            return ToolError(error=f"Failed to read artifact: {e}")
