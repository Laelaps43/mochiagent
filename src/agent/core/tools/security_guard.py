"""
Tool security guard.

Single-entry guard with:
- PathChecks (x-workspace-path / x-workspace-cwd)
- CommandChecks (x-shell-command)
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Mapping, Optional, Set

from pydantic import BaseModel, ConfigDict


class SecurityDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed: bool
    reason: str


class ToolSecurityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enforce_workspace: bool = True
    enforce_command_guard: bool = True
    command_deny_tokens: Optional[Set[str]] = None


class ToolSecurityGuard:
    def __init__(self, root: str | Path, restrict: bool, config: ToolSecurityConfig):
        self.root = Path(root).resolve(strict=False)
        self.restrict = restrict
        self.config = config

    def validate_tool_call(self, tool: Any, arguments: Mapping[str, Any]) -> SecurityDecision:
        schema = getattr(tool, "parameters_schema", {}) or {}
        properties = schema.get("properties", {}) or {}

        for key, prop in properties.items():
            if not isinstance(prop, dict):
                continue
            value = arguments.get(key)
            if value is None:
                continue

            if (prop.get("x-workspace-path") or prop.get("x-workspace-cwd")) and isinstance(
                value, str
            ):
                decision = self._validate_path(value)
                if not decision.allowed:
                    return decision

            if prop.get("x-shell-command") and isinstance(value, str):
                decision = self._validate_shell_command(value, arguments)
                if not decision.allowed:
                    return decision

        return SecurityDecision(allowed=True, reason="allowed")

    def _validate_path(self, raw_path: str) -> SecurityDecision:
        if not self.restrict or not self.config.enforce_workspace:
            return SecurityDecision(allowed=True, reason="workspace guard disabled")

        normalized = self._normalize_path(raw_path)
        if not self._is_inside_root(normalized):
            return SecurityDecision(
                allowed=False,
                reason=f"path '{raw_path}' resolves to '{normalized}' outside workspace root '{self.root}'",
            )
        return SecurityDecision(allowed=True, reason="path allowed")

    def _validate_shell_command(
        self,
        command: str,
        arguments: Mapping[str, Any],
    ) -> SecurityDecision:
        if not self.config.enforce_command_guard:
            return SecurityDecision(allowed=True, reason="command guard disabled")

        deny_tokens = self.config.command_deny_tokens or set()
        if deny_tokens:
            try:
                cmd_tokens = shlex.split(command, posix=True)
            except ValueError:
                cmd_tokens = command.split()
            for deny_token in deny_tokens:
                if deny_token in cmd_tokens:
                    return SecurityDecision(
                        allowed=False,
                        reason=f"command contains denied token: {deny_token!r}",
                    )

        if not self.restrict or not self.config.enforce_workspace:
            return SecurityDecision(allowed=True, reason="workspace guard disabled")

        cwd_raw = (
            arguments.get("workdir") or arguments.get("cwd") or arguments.get("working_directory")
        )
        cwd = (
            self._normalize_path(str(cwd_raw))
            if isinstance(cwd_raw, str) and cwd_raw.strip()
            else self.root
        )

        if not self._is_inside_root(cwd):
            return SecurityDecision(
                allowed=False,
                reason=f"workdir '{cwd_raw}' resolves outside workspace root",
            )

        paths = self._extract_paths_from_command(command)
        for raw in paths:
            normalized = self._normalize_path(raw, cwd=cwd)
            if not self._is_inside_root(normalized):
                return SecurityDecision(
                    allowed=False,
                    reason=f"command path '{raw}' resolves to '{normalized}' outside workspace root '{self.root}'",
                )

        return SecurityDecision(allowed=True, reason="command allowed")

    @staticmethod
    def _extract_paths_from_command(command: str) -> list[str]:
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError:
            # Unbalanced quotes etc. Fail closed on malformed command.
            return ["__INVALID_COMMAND__"]

        path_candidates: list[str] = []
        for token in tokens:
            if token == "__INVALID_COMMAND__":
                path_candidates.append(token)
                continue
            if token.startswith("-"):
                continue
            if token.startswith("/") or token.startswith("~/"):
                path_candidates.append(token)
                continue
            if token.startswith("./") or token.startswith("../") or token in {".", ".."}:
                path_candidates.append(token)
                continue
            if "/" in token:
                path_candidates.append(token)

        return path_candidates

    def _normalize_path(self, raw_path: str, cwd: Path | None = None) -> Path:
        if raw_path == "__INVALID_COMMAND__":
            return self.root.parent / "__INVALID_COMMAND__"

        base = cwd or self.root
        text = raw_path.strip()
        if text.startswith("~/"):
            candidate = Path.home() / text[2:]
        else:
            candidate = Path(text)
            if not candidate.is_absolute():
                candidate = base / candidate
        return candidate.resolve(strict=False)

    def _is_inside_root(self, path: Path) -> bool:
        try:
            path.relative_to(self.root)
            return True
        except ValueError:
            return False
