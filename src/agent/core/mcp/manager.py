"""Agent-scoped MCP runtime state management."""

from __future__ import annotations

import asyncio
import json
import random
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx
from loguru import logger

from agent.core.tools import ToolRegistry
from agent.core.utils import format_exception, to_int
from .types import MCPServerConfig, MCPServerSnapshot, MCPServerSnapshotConfig, MCPServerState

if TYPE_CHECKING:
    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
    from mcp import ClientSession
    from mcp.shared.message import SessionMessage

    _ReadStream = MemoryObjectReceiveStream[SessionMessage | Exception]
    _WriteStream = MemoryObjectSendStream[SessionMessage]


class MCPManager:
    """Agent-scoped MCP runtime state manager."""

    def __init__(self, registry: ToolRegistry, default_timeout: int = 30) -> None:
        self._registry: ToolRegistry = registry
        self._default_timeout: int = default_timeout
        self._states: dict[str, MCPServerState] = {}
        self._configs: dict[str, MCPServerConfig] = {}
        self._sessions: dict[str, ClientSession] = {}
        self._server_stacks: dict[str, AsyncExitStack] = {}
        self._server_tool_names: dict[str, list[str]] = {}
        self._reconnect_locks: dict[str, asyncio.Lock] = {}

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @staticmethod
    def load_config(path: Path) -> dict[str, MCPServerConfig]:
        """Load MCP server configs from a JSON file.

        Returns:
            Dict mapping server names to configs (empty if file missing or invalid).
        """
        if not path.exists():
            logger.info("MCP config not found: {}", path)
            return {}
        try:
            data = cast(object, json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.error("Failed to read MCP config '{}': {}", path, exc)
            return {}

        if not isinstance(data, dict):
            logger.info("MCP config is not a JSON object in {}", path)
            return {}

        payload = cast(dict[str, object], data)
        raw_servers: object = payload.get("mcpServers")
        if not isinstance(raw_servers, dict) or not raw_servers:
            logger.info("No mcpServers configured in {}", path)
            return {}

        servers_map = cast(dict[str, object], raw_servers)
        servers: dict[str, MCPServerConfig] = {}
        for name, raw in servers_map.items():
            if isinstance(raw, MCPServerConfig):
                servers[name] = raw
            elif isinstance(raw, dict):
                try:
                    servers[name] = MCPServerConfig.model_validate(raw)
                except Exception as exc:
                    logger.warning("Invalid MCP server config '{}': {}", name, exc)
            else:
                logger.warning(
                    "MCP server config '{}' is not a dict, skipped",
                    name,
                )
        return servers

    def register_server(self, server_name: str, cfg: MCPServerConfig) -> MCPServerState:
        state = self._states.get(server_name) or MCPServerState()
        state.connect_timeout_ms = to_int(cfg.connectTimeoutMs, default=8000, minimum=100)
        state.max_retries = to_int(cfg.maxRetries, default=2, minimum=0)
        state.retry_initial_ms = to_int(cfg.retryInitialMs, default=300, minimum=50)
        state.retry_max_ms = to_int(cfg.retryMaxMs, default=3000, minimum=100)
        if state.retry_max_ms < state.retry_initial_ms:
            state.retry_max_ms = state.retry_initial_ms
        state.failure_threshold = to_int(cfg.failureThreshold, default=3, minimum=1)
        state.cooldown_sec = to_int(cfg.cooldownSec, default=20, minimum=1)
        state.tool_timeout_sec = to_int(cfg.toolTimeout, default=self._default_timeout, minimum=1)
        self._states[server_name] = state
        return state

    def snapshot(self) -> dict[str, MCPServerSnapshot]:
        result: dict[str, MCPServerSnapshot] = {}
        for name, state in self._states.items():
            data = state.model_dump()
            snap = MCPServerSnapshot.model_validate(data)
            snap.config = MCPServerSnapshotConfig.model_validate(data)
            result[name] = snap
        return result

    def set_status(self, server_name: str, status: str) -> None:
        state = self._states.get(server_name)
        if state is None:
            return
        if state.status != status:
            logger.info(
                "MCP server '{}' status changed: {} -> {}",
                server_name,
                state.status,
                status,
            )
            state.status = status

    def can_execute(self, server_name: str) -> bool:
        state = self._states.get(server_name)
        if state is None:
            return True
        if state.status != "failed":
            return True
        if state.next_retry_at is None:
            return False
        if time.time() >= state.next_retry_at:
            state.next_retry_at = None
            self.set_status(server_name, "degraded")
            return True
        return False

    def record_tool_success(self, server_name: str) -> None:
        state = self._states.get(server_name)
        if state is None:
            return
        state.consecutive_failures = 0
        state.last_error = None
        state.next_retry_at = None
        self.set_status(server_name, "connected")

    def record_tool_failure(self, server_name: str, reason: str) -> None:
        state = self._states.get(server_name)
        if state is None:
            return

        state.consecutive_failures += 1
        state.last_error = reason

        if state.consecutive_failures >= state.failure_threshold:
            _ = self._sessions.pop(server_name, None)
            state.next_retry_at = time.time() + state.cooldown_sec
            self.set_status(server_name, "failed")
            logger.warning(
                "MCP server '{}' reached failure threshold ({}/{}), enter cooldown {}s",
                server_name,
                state.consecutive_failures,
                state.failure_threshold,
                state.cooldown_sec,
            )
            return

        self.set_status(server_name, "degraded")

    def get_session(self, server_name: str) -> ClientSession | None:
        return self._sessions.get(server_name)

    async def reconnect_server(self, server_name: str) -> bool:
        """Close old connection and rebuild for a single server. Returns True on success."""
        lock = self._reconnect_locks.setdefault(server_name, asyncio.Lock())
        if lock.locked():
            # Another coroutine is already reconnecting; wait and check result.
            async with lock:
                return self._sessions.get(server_name) is not None

        async with lock:
            cfg = self._configs.get(server_name)
            if cfg is None:
                return False
            # Tear down old stack
            old_stack = self._server_stacks.pop(server_name, None)
            if old_stack:
                try:
                    await old_stack.aclose()
                except Exception:
                    pass
            _ = self._sessions.pop(server_name, None)
            self._unregister_server_tools(server_name)
            # Reconnect
            state = self._states.get(server_name) or self.register_server(server_name, cfg)
            count = await self._connect_single_server(server_name, cfg, state)
            return count > 0

    def _unregister_server_tools(self, server_name: str) -> None:
        for name in self._server_tool_names.pop(server_name, []):
            self._registry.unregister(name)

    def mark_disconnected(self) -> None:
        for state in self._states.values():
            state.tool_count = 0
            state.next_retry_at = None
            state.status = "disconnected"

    async def close(self) -> None:
        """Close all MCP connections and mark servers disconnected."""
        for server_name in list(self._server_stacks):
            stack = self._server_stacks.pop(server_name, None)
            if stack:
                try:
                    await stack.aclose()
                except Exception as exc:
                    logger.warning(
                        "MCP server '{}' stack close raised: {}",
                        server_name,
                        format_exception(exc),
                    )
        self._sessions.clear()
        self.mark_disconnected()

    async def connect_servers(self, servers: dict[str, MCPServerConfig]) -> int:
        """Connect MCP servers and register their tools.

        Each server gets its own ``AsyncExitStack``.  Call :meth:`close` to
        tear down all connections.

        Returns:
            Total number of tools registered.
        """
        registered = 0
        for server_name, cfg in servers.items():
            self._configs[server_name] = cfg
            state = self.register_server(server_name, cfg)
            registered += await self._connect_single_server(server_name, cfg, state)

        return registered

    async def _connect_single_server(
        self,
        server_name: str,
        cfg: MCPServerConfig,
        state: MCPServerState,
    ) -> int:
        """Connect one MCP server with retries. Returns number of tools registered."""
        from mcp import ClientSession as _ClientSession

        # Delay import to avoid module cycle at import time.
        from agent.common.tools.mcp import MCPToolWrapper

        self.set_status(server_name, "connecting")

        if not cfg.command and not cfg.url:
            state.last_error = "missing command/url"
            state.tool_count = 0
            self.set_status(server_name, "failed")
            logger.warning("MCP server '{}' has no command/url, skipped", server_name)
            return 0

        attempts = state.max_retries + 1
        delay_ms = state.retry_initial_ms
        timeout_sec = state.connect_timeout_ms / 1000

        for attempt in range(1, attempts + 1):
            attempt_stack = AsyncExitStack()
            _ = await attempt_stack.__aenter__()
            attempt_ok = False
            try:
                read, write = await self._open_transport(cfg, attempt_stack, timeout_sec)

                session = await asyncio.wait_for(
                    attempt_stack.enter_async_context(_ClientSession(read, write)),
                    timeout=timeout_sec,
                )
                _ = await asyncio.wait_for(session.initialize(), timeout=timeout_sec)
                listed = await asyncio.wait_for(session.list_tools(), timeout=timeout_sec)

                tool_names: list[str] = []
                for tool_def in listed.tools:
                    wrapper = MCPToolWrapper(
                        server_name=server_name,
                        tool_def=tool_def,
                        timeout=state.tool_timeout_sec,
                        manager=self,
                    )
                    self._registry.register(wrapper)
                    tool_names.append(wrapper.name)

                state.tool_count = len(listed.tools)
                state.last_error = None
                state.consecutive_failures = 0
                state.last_connected_at = time.time()
                state.next_retry_at = None
                self.set_status(server_name, "connected")
                logger.info(
                    "MCP server '{}' connected, registered {} tools",
                    server_name,
                    state.tool_count,
                )
                committed_stack = attempt_stack.pop_all()
                self._sessions[server_name] = session
                self._server_stacks[server_name] = committed_stack
                self._server_tool_names[server_name] = tool_names
                attempt_ok = True
                return state.tool_count

            except Exception as exc:
                state.last_error = format_exception(exc)
                if attempt >= attempts:
                    state.tool_count = 0
                    self.set_status(server_name, "failed")
                    logger.error(
                        "MCP server '{}' connect failed after {}/{} attempts: {}",
                        server_name,
                        attempt,
                        attempts,
                        state.last_error,
                    )
                    return 0

                wait_ms = min(delay_ms, state.retry_max_ms)
                jitter_ms = int(wait_ms * 0.2 * random.random())
                total_wait_ms = wait_ms + jitter_ms
                state.next_retry_at = time.time() + (total_wait_ms / 1000)
                logger.warning(
                    "MCP server '{}' connect attempt {}/{} failed: {}. retry in {}ms",
                    server_name,
                    attempt,
                    attempts,
                    state.last_error,
                    total_wait_ms,
                )
                await asyncio.sleep(total_wait_ms / 1000)
                delay_ms = min(delay_ms * 2, state.retry_max_ms)
            finally:
                if not attempt_ok:
                    try:
                        await attempt_stack.aclose()
                    except Exception as close_exc:
                        logger.warning(
                            "MCP server '{}' cleanup after failed attempt raised: {}",
                            server_name,
                            format_exception(close_exc),
                        )
        return 0

    @staticmethod
    async def _open_transport(
        cfg: MCPServerConfig,
        stack: AsyncExitStack,
        timeout_sec: float,
    ) -> tuple[_ReadStream, _WriteStream]:
        """Open stdio or remote transport. Returns ``(read_stream, write_stream)``."""
        from mcp import StdioServerParameters
        from mcp.client.sse import sse_client
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamable_http_client

        if cfg.command:
            params = StdioServerParameters(
                command=cfg.command,
                args=cfg.args or [],
                env=cfg.env or None,
            )
            return await asyncio.wait_for(
                stack.enter_async_context(stdio_client(params)),
                timeout=timeout_sec,
            )

        # Remote transport: try streamable_http first, fall back to SSE.
        remote_url = str(cfg.url)
        headers = cfg.headers or None
        transport_order = (
            ["sse", "streamable_http"]
            if remote_url.rstrip("/").endswith("/sse")
            else ["streamable_http", "sse"]
        )
        transport_errors: list[Exception] = []

        for mode in transport_order:
            try:
                if mode == "streamable_http":
                    if headers:
                        http_client = await stack.enter_async_context(
                            httpx.AsyncClient(headers=headers, follow_redirects=True)
                        )
                        read, write, _ = await asyncio.wait_for(
                            stack.enter_async_context(
                                streamable_http_client(remote_url, http_client=http_client)
                            ),
                            timeout=timeout_sec,
                        )
                    else:
                        read, write, _ = await asyncio.wait_for(
                            stack.enter_async_context(streamable_http_client(remote_url)),
                            timeout=timeout_sec,
                        )
                else:
                    read, write = await asyncio.wait_for(
                        stack.enter_async_context(sse_client(remote_url, headers=headers)),
                        timeout=timeout_sec,
                    )

                logger.info("MCP remote transport selected: '{}'", mode)
                return read, write
            except Exception as exc:
                transport_errors.append(exc)

        if len(transport_errors) == 1:
            raise transport_errors[0]
        raise ExceptionGroup(
            f"remote transport attempts failed for {remote_url}",
            transport_errors,
        )
