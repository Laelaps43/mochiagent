"""Sandbox types — leaf module (only depends on pydantic/typing)."""

from __future__ import annotations

from typing import ClassVar, Literal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class SandboxDecision(BaseModel):
    """Immutable decision returned by every sandbox check."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    allowed: bool
    reason: str


class SandboxConfig(BaseModel):
    """Unified sandbox configuration.

    Replaces ``WorkspaceConfig`` and ``ToolSecurityConfig``.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    backend: Literal["noop", "seatbelt", "bwrap", "docker"] = "noop"
    workspace_root: Path = Field(default_factory=Path.cwd)
    restrict_to_workspace: bool = True
    network: bool = False

    # Application-level command filtering (used by NoopSandbox; OS-level
    # backends inherit these checks and add their own enforcement).
    command_deny_tokens: set[str] = Field(
        default_factory=lambda: {
            "`",
            "$(",
            "\n",
            "\r",
            ";",
            "&&",
            "||",
            "|",
            ">",
            ">>",
            "<",
        },
    )

    # Opaque dict for backend-specific options (seatbelt profile path,
    # bwrap flags, docker image, etc.).
    backend_options: dict[str, object] = Field(default_factory=dict)
