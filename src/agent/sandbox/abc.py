"""Sandbox abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from agent.sandbox.types import SandboxConfig, SandboxDecision

if TYPE_CHECKING:
    from agent.core.tools.base import Tool


class Sandbox(ABC):
    """Pluggable sandbox interface.

    Every method is async so implementations are free to delegate to
    external processes or services if they need to.
    """

    def __init__(self, config: SandboxConfig) -> None:
        self._config: SandboxConfig = config

    @property
    def config(self) -> SandboxConfig:
        return self._config

    @property
    def workspace_root(self) -> Path:
        return self._config.workspace_root

    # ------------------------------------------------------------------
    # 1. Pre-execution validation (called by ToolExecutor)
    # ------------------------------------------------------------------

    @abstractmethod
    async def validate_tool_call(
        self,
        tool: object,
        arguments: Mapping[str, object],
    ) -> SandboxDecision:
        """Decide whether the tool call is permitted.

        Reads the tool's ``parameters_schema`` for ``x-workspace-path``,
        ``x-shell-command``, etc. and delegates to :meth:`check_path` /
        command-check helpers.
        """
        ...

    # ------------------------------------------------------------------
    # 2. Path checking (called by individual file tools)
    # ------------------------------------------------------------------

    @abstractmethod
    async def check_path(
        self,
        raw_path: str,
        mode: Literal["read", "write"] = "read",
    ) -> SandboxDecision:
        """Check whether *raw_path* is accessible within the sandbox."""
        ...

    # ------------------------------------------------------------------
    # 3. Sandboxed command execution (called by ExecTool)
    # ------------------------------------------------------------------

    @abstractmethod
    async def exec_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> tuple[int | None, str, bool]:
        """Execute a command within the sandbox.

        Returns ``(exit_code, combined_output, truncated)``.
        """
        ...

    # ------------------------------------------------------------------
    # 4. Tool execution (called by ToolExecutor)
    # ------------------------------------------------------------------

    async def run_tool(self, tool: Tool, arguments: dict[str, object]) -> object:
        """Execute a tool within this sandbox environment.

        The default implementation runs the tool in-process and manages
        the sandbox context variable so that ``get_sandbox()`` works
        inside the tool.  OS-level backends override this to spawn a
        worker subprocess for ``"subprocess"``-mode tools.
        """
        from agent.sandbox.context import reset_sandbox, set_sandbox

        set_sandbox(self)
        try:
            return await tool.execute(**arguments)
        finally:
            reset_sandbox()
