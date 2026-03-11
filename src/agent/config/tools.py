"""Tool runtime config exposed to package users."""

from __future__ import annotations

from pathlib import Path
from typing import Set

from pydantic import BaseModel, ConfigDict, Field


class ToolPolicyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    allow: Set[str] = Field(default_factory=set)
    deny: Set[str] = Field(default_factory=set)

    @classmethod
    def from_csv(cls, *, allow_csv: str = "", deny_csv: str = "") -> "ToolPolicyConfig":
        def parse(raw: str) -> set[str]:
            if not raw:
                return set()
            return {item.strip() for item in raw.split(",") if item and item.strip()}

        return cls(allow=parse(allow_csv), deny=parse(deny_csv))


class WorkspaceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    root: Path = Field(default_factory=Path.cwd)
    restrict: bool = True


class ToolSecurityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enforce_workspace: bool = True
    enforce_command_guard: bool = True
    command_deny_tokens: Set[str] = Field(default_factory=lambda: {"`", "$(", "\n", "\r"})


class ToolRuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    timeout: int = 30
    policy: ToolPolicyConfig = Field(default_factory=ToolPolicyConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    security: ToolSecurityConfig = Field(default_factory=ToolSecurityConfig)
    exec_max_output_chars: int = 20000
    web_fetch_max_chars: int = 20000
    web_search_api_key: str = ""
