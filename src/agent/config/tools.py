"""Tool runtime config exposed to package users."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ToolPolicyConfig:
    allow: set[str] = field(default_factory=set)
    deny: set[str] = field(default_factory=set)

    @classmethod
    def from_csv(cls, *, allow_csv: str = "", deny_csv: str = "") -> "ToolPolicyConfig":
        def parse(raw: str) -> set[str]:
            if not raw:
                return set()
            return {item.strip() for item in raw.split(",") if item and item.strip()}

        return cls(allow=parse(allow_csv), deny=parse(deny_csv))


@dataclass(frozen=True)
class WorkspaceConfig:
    root: Path = field(default_factory=Path.cwd)
    restrict: bool = True


@dataclass(frozen=True)
class ToolSecurityConfig:
    enforce_workspace: bool = True
    enforce_command_guard: bool = True
    command_deny_tokens: set[str] = field(default_factory=lambda: {"`", "$(", "\n", "\r"})


@dataclass(frozen=True)
class ToolRuntimeConfig:
    timeout: int = 30
    policy: ToolPolicyConfig = field(default_factory=ToolPolicyConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    security: ToolSecurityConfig = field(default_factory=ToolSecurityConfig)
    exec_max_output_chars: int = 20000
    web_fetch_max_chars: int = 20000
    web_search_api_key: str = ""
