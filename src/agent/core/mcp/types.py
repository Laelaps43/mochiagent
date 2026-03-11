"""Type definitions for MCP runtime manager."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class MCPServerState(BaseModel):
    status: str = "disconnected"  # connecting|connected|degraded|failed|disconnected
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    last_connected_at: Optional[float] = None
    tool_count: int = 0
    next_retry_at: Optional[float] = None
    connect_timeout_ms: int = 8000
    max_retries: int = 2
    retry_initial_ms: int = 300
    retry_max_ms: int = 3000
    failure_threshold: int = 3
    cooldown_sec: int = 20
    tool_timeout_sec: int = 30


class MCPServerConfig(BaseModel):
    command: str = ""
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    url: str = ""
    headers: Dict[str, str] = Field(default_factory=dict)
    connectTimeoutMs: Optional[int] = None
    maxRetries: Optional[int] = None
    retryInitialMs: Optional[int] = None
    retryMaxMs: Optional[int] = None
    failureThreshold: Optional[int] = None
    cooldownSec: Optional[int] = None
    toolTimeout: Optional[int] = None


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
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    last_connected_at: Optional[float] = None
    tool_count: int = 0
    next_retry_at: Optional[float] = None
    config: MCPServerSnapshotConfig = Field(default_factory=MCPServerSnapshotConfig)
