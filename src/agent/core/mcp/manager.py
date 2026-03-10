"""Agent-scoped MCP runtime state management."""

from __future__ import annotations

import asyncio
import random
import time
from contextlib import AsyncExitStack
from typing import Any, Mapping

import httpx
from loguru import logger

from agent.core.tools import ToolRegistry
from .types import MCPServerConfig, MCPServerSnapshot, MCPServerState


def _to_int(value: Any, default: int, minimum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, parsed)


def _collect_exception_messages(exc: BaseException, out: list[str]) -> None:
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            _collect_exception_messages(sub, out)
        return
    out.append(f"{type(exc).__name__}: {exc}")


def _format_exception(exc: BaseException) -> str:
    messages: list[str] = []
    _collect_exception_messages(exc, messages)
    if not messages:
        return f"{type(exc).__name__}: {exc}"
    unique: list[str] = []
    for msg in messages:
        if msg not in unique:
            unique.append(msg)
    if len(unique) <= 3:
        return " | ".join(unique)
    return " | ".join(unique[:3]) + f" | ... (+{len(unique) - 3} more)"


class MCPManager:
    """Agent-scoped MCP runtime state manager."""

    def __init__(self, registry: ToolRegistry, default_timeout: int = 30):
        self._registry = registry
        self._default_timeout = default_timeout
        self._states: dict[str, MCPServerState] = {}

    @staticmethod
    def _coerce_server_config(raw: object) -> MCPServerConfig:
        if isinstance(raw, dict):
            return raw
        logger.warning(
            "MCP server config is not a dict (got {}), using empty config",
            type(raw).__name__,
        )
        return {}

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    def register_server(self, server_name: str, cfg: MCPServerConfig) -> MCPServerState:
        state = self._states.get(server_name) or MCPServerState()
        state.connect_timeout_ms = _to_int(cfg.get("connectTimeoutMs"), default=8000, minimum=100)
        state.max_retries = _to_int(cfg.get("maxRetries"), default=2, minimum=0)
        state.retry_initial_ms = _to_int(cfg.get("retryInitialMs"), default=300, minimum=50)
        state.retry_max_ms = _to_int(cfg.get("retryMaxMs"), default=3000, minimum=100)
        if state.retry_max_ms < state.retry_initial_ms:
            state.retry_max_ms = state.retry_initial_ms
        state.failure_threshold = _to_int(cfg.get("failureThreshold"), default=3, minimum=1)
        state.cooldown_sec = _to_int(cfg.get("cooldownSec"), default=20, minimum=1)
        state.tool_timeout_sec = _to_int(
            cfg.get("toolTimeout"), default=self._default_timeout, minimum=1
        )
        self._states[server_name] = state
        return state

    def snapshot(self) -> dict[str, MCPServerSnapshot]:
        return {
            name: {
                "status": state.status,
                "last_error": state.last_error,
                "consecutive_failures": state.consecutive_failures,
                "last_connected_at": state.last_connected_at,
                "tool_count": state.tool_count,
                "next_retry_at": state.next_retry_at,
                "config": {
                    "connect_timeout_ms": state.connect_timeout_ms,
                    "max_retries": state.max_retries,
                    "retry_initial_ms": state.retry_initial_ms,
                    "retry_max_ms": state.retry_max_ms,
                    "failure_threshold": state.failure_threshold,
                    "cooldown_sec": state.cooldown_sec,
                    "tool_timeout_sec": state.tool_timeout_sec,
                },
            }
            for name, state in self._states.items()
        }

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

    def mark_disconnected(self) -> None:
        for state in self._states.values():
            state.tool_count = 0
            state.next_retry_at = None
            state.status = "disconnected"

    async def connect_servers(
        self,
        *,
        mcp_servers: Mapping[str, object],
        stack: AsyncExitStack,
    ) -> int:
        """
        Connect MCP servers and register their tools.

        Returns:
            Total number of tools registered.
        """
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.sse import sse_client
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamable_http_client
        except Exception as exc:
            raise RuntimeError(
                "MCP support requires python package 'mcp'. Please install dependencies."
            ) from exc

        # Delay import to avoid module cycle at import time.
        from agent.common.tools.mcp import MCPToolWrapper

        registered = 0
        for server_name, raw_cfg in mcp_servers.items():
            cfg = self._coerce_server_config(raw_cfg)
            state = self.register_server(server_name, cfg)
            self.set_status(server_name, "connecting")

            attempts = state.max_retries + 1
            attempt = 0
            delay_ms = state.retry_initial_ms
            connected = False

            while attempt < attempts and not connected:
                attempt += 1
                attempt_stack = AsyncExitStack()
                await attempt_stack.__aenter__()
                attempt_ok = False
                try:
                    if cfg.get("command"):
                        params = StdioServerParameters(
                            command=cfg["command"],
                            args=cfg.get("args", []),
                            env=cfg.get("env") or None,
                        )
                        read, write = await asyncio.wait_for(
                            attempt_stack.enter_async_context(stdio_client(params)),
                            timeout=state.connect_timeout_ms / 1000,
                        )
                    elif cfg.get("url"):
                        remote_url = str(cfg["url"])
                        headers = cfg.get("headers") or None
                        read = None
                        write = None
                        transport_errors: list[BaseException] = []
                        transport_order = (
                            ["sse", "streamable_http"]
                            if remote_url.rstrip("/").endswith("/sse")
                            else ["streamable_http", "sse"]
                        )

                        for transport_mode in transport_order:
                            try:
                                if transport_mode == "streamable_http":
                                    if headers:
                                        http_client = await attempt_stack.enter_async_context(
                                            httpx.AsyncClient(
                                                headers=headers,
                                                follow_redirects=True,
                                            )
                                        )
                                        read, write, _ = await asyncio.wait_for(
                                            attempt_stack.enter_async_context(
                                                streamable_http_client(
                                                    remote_url,
                                                    http_client=http_client,
                                                )
                                            ),
                                            timeout=state.connect_timeout_ms / 1000,
                                        )
                                    else:
                                        read, write, _ = await asyncio.wait_for(
                                            attempt_stack.enter_async_context(
                                                streamable_http_client(remote_url)
                                            ),
                                            timeout=state.connect_timeout_ms / 1000,
                                        )
                                else:
                                    read, write = await asyncio.wait_for(
                                        attempt_stack.enter_async_context(
                                            sse_client(remote_url, headers=headers)
                                        ),
                                        timeout=state.connect_timeout_ms / 1000,
                                    )

                                logger.info(
                                    "MCP server '{}' selected transport '{}'",
                                    server_name,
                                    transport_mode,
                                )
                                break
                            except Exception as transport_exc:
                                transport_errors.append(transport_exc)
                                read = None
                                write = None

                        if read is None or write is None:
                            if transport_errors:
                                if len(transport_errors) == 1:
                                    raise transport_errors[0]
                                raise ExceptionGroup(
                                    f"remote transport attempts failed for {remote_url}",
                                    transport_errors,
                                )
                            raise RuntimeError(
                                f"failed to create remote transport for {remote_url}"
                            )
                    else:
                        state.last_error = "missing command/url"
                        state.tool_count = 0
                        self.set_status(server_name, "failed")
                        logger.warning(
                            "MCP server '{}' has no command/url, skipped",
                            server_name,
                        )
                        break

                    session = await asyncio.wait_for(
                        attempt_stack.enter_async_context(ClientSession(read, write)),
                        timeout=state.connect_timeout_ms / 1000,
                    )
                    await asyncio.wait_for(
                        session.initialize(),
                        timeout=state.connect_timeout_ms / 1000,
                    )
                    listed = await asyncio.wait_for(
                        session.list_tools(),
                        timeout=state.connect_timeout_ms / 1000,
                    )

                    for tool_def in listed.tools:
                        wrapper = MCPToolWrapper(
                            session=session,
                            server_name=server_name,
                            tool_def=tool_def,
                            timeout=state.tool_timeout_sec,
                            manager=self,
                        )
                        self._registry.register(wrapper)
                        registered += 1

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
                    stack.push_async_callback(committed_stack.aclose)
                    attempt_ok = True
                    connected = True
                except Exception as exc:
                    state.last_error = _format_exception(exc)
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
                        break

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
                                _format_exception(close_exc),
                            )

        return registered
