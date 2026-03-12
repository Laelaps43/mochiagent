"""Type definitions for MCP runtime manager."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MCPServerState(BaseModel):
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


class MCPServerConfig(BaseModel):
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    connectTimeoutMs: int | None = None
    maxRetries: int | None = None
    retryInitialMs: int | None = None
    retryMaxMs: int | None = None
    failureThreshold: int | None = None
    cooldownSec: int | None = None
    toolTimeout: int | None = None


class MCPServerSnapshotConfig(BaseModel):
    connect_timeout_ms: int = 0
    max_retries: int = 0
    retry_initial_ms: int = 0
    retry_max_ms: int = 0
    failure_threshold: int = 0
    cooldown_sec: int = 0
    tool_timeout_sec: int = 0


class MCPServerSnapshot(BaseModel):
    status: str = ""
    last_error: str | None = None
    consecutive_failures: int = 0
    last_connected_at: float | None = None
    tool_count: int = 0
    next_retry_at: float | None = None
    config: MCPServerSnapshotConfig = Field(default_factory=MCPServerSnapshotConfig)
