from __future__ import annotations

import asyncio
import re
from typing import Protocol, override

from mcp import ClientSession
from mcp.types import Tool as MCPToolDef, TextContent

from agent.core.tools import Tool


class _MCPManagerProtocol(Protocol):
    def can_execute(self, server_name: str) -> bool: ...
    def record_tool_success(self, server_name: str) -> None: ...
    def record_tool_failure(self, server_name: str, reason: str) -> None: ...


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", value)


class MCPToolWrapper(Tool):
    def __init__(
        self,
        session: ClientSession,
        server_name: str,
        tool_def: MCPToolDef,
        timeout: int,
        manager: _MCPManagerProtocol | None = None,
    ):
        self._session: ClientSession = session
        self._server_name: str = server_name
        self._original_name: str = tool_def.name
        self._name: str = f"mcp_{_safe_name(server_name)}_{_safe_name(tool_def.name)}"
        self._description: str = tool_def.description or tool_def.name
        self._schema: dict[str, object] = tool_def.inputSchema or {
            "type": "object",
            "properties": {},
        }
        self._timeout: int = timeout
        self._manager: _MCPManagerProtocol | None = manager

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def description(self) -> str:
        return self._description

    @property
    @override
    def parameters_schema(self) -> dict[str, object]:
        return self._schema

    @override
    async def execute(self, **kwargs: object) -> str:
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
            if isinstance(block, TextContent):
                parts.append(block.text)
            else:
                parts.append(str(block))

        structured_content = (
            result.structuredContent if hasattr(result, "structuredContent") else None
        )
        if not parts and structured_content is not None:
            parts.append(str(result.structuredContent))

        return "\n".join(parts).strip() or "(no output)"
