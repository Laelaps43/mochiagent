"""BwrapSandbox — Linux bubblewrap wrapper.

Inherits application-level checks from :class:`NoopSandbox` and wraps
command execution with ``bwrap`` to create a restricted namespace.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from typing import TYPE_CHECKING, override

from loguru import logger

from agent.sandbox.backends.noop import NoopSandbox
from agent.sandbox.protocol import WorkerRequest, WorkerResponse
from agent.sandbox.types import SandboxConfig

if TYPE_CHECKING:
    from agent.core.tools.base import Tool

_BWRAP_BIN = "bwrap"

# System directories mounted read-only inside the sandbox.
_RO_SYSTEM_DIRS: list[str] = [
    "/bin",
    "/sbin",
    "/usr",
    "/lib",
    "/lib32",
    "/lib64",
    "/etc",
    "/nix/store",
]


def _build_bwrap_args(
    command: str,
    workspace: str,
    *,
    network: bool = False,
) -> list[str]:
    """Build the full ``bwrap`` argument list."""
    args = [_BWRAP_BIN]

    # Unshare all namespaces
    args.append("--unshare-all")
    args.append("--new-session")
    args.append("--die-with-parent")

    # Read-only root filesystem
    args.extend(["--ro-bind", "/", "/"])

    # Essential pseudo-filesystems
    args.extend(["--proc", "/proc"])
    args.extend(["--dev", "/dev"])
    args.extend(["--tmpfs", "/tmp"])

    # Workspace: read-write bind
    ws = str(workspace)
    args.extend(["--bind", ws, ws])

    # Optionally share network namespace
    if network:
        args.append("--share-net")

    # Execute command
    args.append("--")
    args.extend(["sh", "-c", command])

    return args


class BwrapSandbox(NoopSandbox):
    """Linux bubblewrap sandbox.

    Falls back to :class:`NoopSandbox` if ``bwrap`` is not installed.
    """

    def __init__(self, config: SandboxConfig) -> None:
        super().__init__(config)
        if not shutil.which(_BWRAP_BIN):
            logger.warning(
                "bwrap not found in PATH; BwrapSandbox will run commands without OS-level isolation"
            )
            self._available: bool = False
        else:
            self._available = True

    # ------------------------------------------------------------------
    # run_tool — execute tool in sandboxed worker subprocess
    # ------------------------------------------------------------------

    @override
    async def run_tool(self, tool: Tool, arguments: dict[str, object]) -> object:
        if tool.sandbox_mode == "inprocess" or not self._available:
            return await super().run_tool(tool, arguments)

        tool_cls = type(tool)
        request = WorkerRequest(
            tool_module=tool_cls.__module__,
            tool_class=tool_cls.__name__,
            init_args=tool.serialize_init_args(),
            execute_args=arguments,
            workspace_root=str(self.workspace_root),
        )

        # Build bwrap args wrapping the worker command
        worker_cmd = f"{sys.executable} -m agent.sandbox.worker"
        bwrap_args = _build_bwrap_args(
            worker_cmd,
            workspace=str(self.workspace_root),
            network=self._config.network,
        )

        proc = await asyncio.create_subprocess_exec(
            *bwrap_args,
            cwd=str(self.workspace_root),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        request_bytes = (request.model_dump_json() + "\n").encode()
        stdout_bytes, stderr_bytes = await proc.communicate(input=request_bytes)

        stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
        if not stdout_text:
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Worker produced no output (exit={proc.returncode}): {stderr_text}"
            )

        response = WorkerResponse.model_validate_json(stdout_text)
        if not response.success:
            raise RuntimeError(response.error or "unknown worker error")
        return response.result

    @override
    async def exec_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> tuple[int | None, str, bool]:
        if not self._available:
            return await super().exec_command(command, cwd=cwd, timeout=timeout)

        import asyncio

        bwrap_args = _build_bwrap_args(
            command,
            workspace=str(self.workspace_root),
            network=self._config.network,
        )

        proc = await asyncio.create_subprocess_exec(
            *bwrap_args,
            cwd=cwd or str(self.workspace_root),
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

        truncated = len(combined) > self._SAFETY_MAX_CHARS
        if truncated:
            combined = combined[: self._SAFETY_MAX_CHARS]

        return proc.returncode, combined, truncated
