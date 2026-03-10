"""Type definitions for MCP runtime manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import TypedDict


@dataclass
class MCPServerState:
    status: str = "disconnected"  # connecting|connected|degraded|failed|disconnected
    last_error: str | None = None
    consecutive_failures: int = 0
    last_connected_at: float | None = None
    tool_count: int = 0
    next_retry_at: float | None = None
    connect_timeout_ms: int = 8000
    max_retries: int = 2
    retry_initial_ms: int = 300
    retry_max_ms: int = 3000
    failure_threshold: int = 3
    cooldown_sec: int = 20
    tool_timeout_sec: int = 30


class MCPServerConfig(TypedDict, total=False):
    command: str
    args: list[str]
    env: dict[str, str]
    url: str
    headers: dict[str, str]
    connectTimeoutMs: int
    maxRetries: int
    retryInitialMs: int
    retryMaxMs: int
    failureThreshold: int
    cooldownSec: int
    toolTimeout: int


class MCPServerSnapshotConfig(TypedDict):
    connect_timeout_ms: int
    max_retries: int
    retry_initial_ms: int
    retry_max_ms: int
    failure_threshold: int
    cooldown_sec: int
    tool_timeout_sec: int


class MCPServerSnapshot(TypedDict):
    status: str
    last_error: str | None
    consecutive_failures: int
    last_connected_at: float | None
    tool_count: int
    next_retry_at: float | None
    config: MCPServerSnapshotConfig
