"""NoopSandbox — application-level checks, no OS-level isolation.

Reproduces the behaviour of the former ``ToolSecurityGuard`` and
``validate_path_within_workspace()``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Literal, override

from agent.sandbox.abc import Sandbox
from agent.sandbox._path_checker import (
    check_command_tokens,
    check_path_access,
    extract_paths_from_command,
    normalize_path,
)
from agent.sandbox._schema_inspector import inspect_tool_call
from agent.sandbox.types import SandboxConfig, SandboxDecision

# Safety cap for subprocess output (1 MB).
_SAFETY_MAX_CHARS = 1024 * 1024


class NoopSandbox(Sandbox):
    _SAFETY_MAX_CHARS: int = _SAFETY_MAX_CHARS
    """Application-level sandbox — path validation + command token filtering.

    This is the default backend and mirrors the pre-sandbox security guard
    behaviour exactly.
    """

    def __init__(self, config: SandboxConfig) -> None:
        super().__init__(config)
        self._deny_tokens: set[str] = config.command_deny_tokens

    # ------------------------------------------------------------------
    # validate_tool_call
    # ------------------------------------------------------------------

    @override
    async def validate_tool_call(
        self,
        tool: object,
        arguments: Mapping[str, object],
    ) -> SandboxDecision:
        return await inspect_tool_call(
            tool,
            arguments,
            check_path=self._check_path_for_tool,
            check_command=self._check_command_for_tool,
        )

    # ------------------------------------------------------------------
    # check_path
    # ------------------------------------------------------------------

    @override
    async def check_path(
        self,
        raw_path: str,
        mode: Literal["read", "write"] = "read",
    ) -> SandboxDecision:
        allowed, reason = check_path_access(
            raw_path,
            root=self.workspace_root,
            restrict=self._config.restrict_to_workspace,
        )
        return SandboxDecision(allowed=allowed, reason=reason)

    # ------------------------------------------------------------------
    # exec_command — bare subprocess (no OS sandbox)
    # ------------------------------------------------------------------

    @override
    async def exec_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> tuple[int | None, str, bool]:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            _ = await proc.wait()
            raise
        except asyncio.CancelledError:
            proc.kill()
            _ = await proc.wait()
            raise

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        combined = stdout + ("\n" if stdout and stderr else "") + stderr

        truncated = len(combined) > _SAFETY_MAX_CHARS
        if truncated:
            combined = combined[:_SAFETY_MAX_CHARS]

        return proc.returncode, combined, truncated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _check_path_for_tool(self, raw_path: str) -> SandboxDecision:
        return await self.check_path(raw_path)

    async def _check_command_for_tool(
        self,
        command: str,
        arguments: Mapping[str, object],
    ) -> SandboxDecision:
        # 1. Token filter
        allowed, reason = check_command_tokens(command, self._deny_tokens)
        if not allowed:
            return SandboxDecision(allowed=False, reason=reason)

        # 2. CWD check
        cwd_raw = (
            arguments.get("workdir")
            or arguments.get("cwd")
            or arguments.get("working_directory")
        )
        cwd = (
            normalize_path(str(cwd_raw), root=self.workspace_root)
            if isinstance(cwd_raw, str) and cwd_raw.strip()
            else self.workspace_root
        )

        if self._config.restrict_to_workspace:
            allowed, reason = check_path_access(
                str(cwd),
                root=self.workspace_root,
                restrict=True,
            )
            if not allowed:
                return SandboxDecision(allowed=False, reason=reason)

        # 3. Paths inside command (including malformed command detection)
        if self._config.restrict_to_workspace:
            extracted = extract_paths_from_command(command)
            if extracted == ["__INVALID_COMMAND__"]:
                return SandboxDecision(allowed=False, reason="malformed command")
            for raw in extracted:
                normalized = normalize_path(raw, root=self.workspace_root, cwd=cwd)
                allowed, reason = check_path_access(
                    str(normalized),
                    root=self.workspace_root,
                    restrict=True,
                    cwd=cwd,
                )
                if not allowed:
                    return SandboxDecision(allowed=False, reason=reason)

        return SandboxDecision(allowed=True, reason="command allowed")
