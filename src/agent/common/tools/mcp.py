"""MCP integration helpers for registering MCP tools into ToolRegistry."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from agent.core.mcp.manager import MCPManager
from agent.core.tools import Tool


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", value)


class MCPToolWrapper(Tool):
    """Wrap one MCP tool as a native Tool."""

    def __init__(
        self,
        session,
        server_name: str,
        tool_def,
        timeout: int,
        manager: MCPManager | None = None,
    ):
        self._session = session
        self._server_name = server_name
        self._original_name = tool_def.name
        self._name = f"mcp_{_safe_name(server_name)}_{_safe_name(tool_def.name)}"
        self._description = tool_def.description or tool_def.name
        self._schema = tool_def.inputSchema or {"type": "object", "properties": {}}
        self._timeout = timeout
        self._manager = manager

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._schema

    async def execute(self, **kwargs: Any) -> str:
        if self._manager and not self._manager.can_execute(self._server_name):
            return f"MCP server '{self._server_name}' is cooling down after failures"

        try:
            result = await asyncio.wait_for(
                self._session.call_tool(self._original_name, arguments=kwargs),
                timeout=self._timeout,
            )
            if self._manager:
                self._manager.record_tool_success(self._server_name)
        except asyncio.TimeoutError:
            if self._manager:
                self._manager.record_tool_failure(self._server_name, "timeout")
            return f"MCP tool timeout after {self._timeout}s"
        except Exception as exc:
            if self._manager:
                self._manager.record_tool_failure(self._server_name, str(exc))
            return f"MCP tool call failed: {exc}"

        parts: list[str] = []
        content_blocks = result.content if hasattr(result, "content") else []
        for block in content_blocks or []:
            text = block.text if hasattr(block, "text") else None
            if isinstance(text, str) and text:
                parts.append(text)
            else:
                parts.append(str(block))

        structured_content = result.structuredContent if hasattr(result, "structuredContent") else None
        if not parts and structured_content is not None:
            parts.append(str(result.structuredContent))

        return "\n".join(parts).strip() or "(no output)"
