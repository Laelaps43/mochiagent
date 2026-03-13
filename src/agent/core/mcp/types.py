"""Type definitions for MCP runtime manager."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


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
    """MCP server configuration.

    Accepts both snake_case (Python) and camelCase (JSON config) field names
    via ``populate_by_name=True`` + ``alias``.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=True)

    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    connect_timeout_ms: int | None = Field(default=None, alias="connectTimeoutMs")
    max_retries: int | None = Field(default=None, alias="maxRetries")
    retry_initial_ms: int | None = Field(default=None, alias="retryInitialMs")
    retry_max_ms: int | None = Field(default=None, alias="retryMaxMs")
    failure_threshold: int | None = Field(default=None, alias="failureThreshold")
    cooldown_sec: int | None = Field(default=None, alias="cooldownSec")
    tool_timeout: int | None = Field(default=None, alias="toolTimeout")


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
