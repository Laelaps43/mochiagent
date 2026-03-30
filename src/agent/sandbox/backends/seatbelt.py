"""SeatbeltSandbox — macOS sandbox-exec wrapper.

Inherits application-level checks from :class:`NoopSandbox` and wraps
command execution with ``/usr/bin/sandbox-exec`` using a dynamically
generated Seatbelt SBPL profile.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, override

from loguru import logger

from agent.sandbox.backends.noop import NoopSandbox
from agent.sandbox.protocol import WorkerRequest, WorkerResponse
from agent.sandbox.types import SandboxConfig

if TYPE_CHECKING:
    from agent.core.tools.base import Tool

_SEATBELT_BIN = "/usr/bin/sandbox-exec"


def _generate_sbpl(
    workspace: Path,
    tmpdir: Path,
    *,
    writable: bool = True,
    network: bool = False,
) -> str:
    """Build a Seatbelt SBPL profile string."""
    ws = str(workspace)
    tmp = str(tmpdir)

    lines = [
        "(version 1)",
        "(deny default)",
        "",
        "; --- process basics ---",
        "(allow process-exec)",
        "(allow process-fork)",
        "(allow signal (target same-sandbox))",
        "(allow process-info* (target same-sandbox))",
        "",
        "; --- system reads ---",
        "(allow file-read* (subpath \"/bin\"))",
        "(allow file-read* (subpath \"/sbin\"))",
        "(allow file-read* (subpath \"/usr\"))",
        "(allow file-read* (subpath \"/lib\"))",
        "(allow file-read* (subpath \"/System\"))",
        "(allow file-read* (subpath \"/etc\"))",
        "(allow file-read* (subpath \"/private/etc\"))",
        "(allow file-read* (subpath \"/dev\"))",
        "(allow sysctl-read)",
        "",
        "; --- workspace ---",
        f"(allow file-read* (subpath \"{ws}\"))",
    ]

    if writable:
        lines.append(f"(allow file-write* (subpath \"{ws}\"))")

    lines += [
        "",
        "; --- tmp ---",
        f"(allow file-read* file-write* (subpath \"{tmp}\"))",
        "",
        "; --- misc ---",
        "(allow file-write-data (require-all (path \"/dev/null\") (vnode-type CHARACTER-DEVICE)))",
        "(allow ipc-posix-sem)",
        "(allow pseudo-tty)",
        "(allow file-read* file-write* file-ioctl (literal \"/dev/ptmx\"))",
    ]

    if network:
        lines += [
            "",
            "; --- network ---",
            "(allow network-outbound)",
            "(allow network-inbound)",
            "(allow system-socket)",
            "(allow mach-lookup)",
        ]

    return "\n".join(lines)


class SeatbeltSandbox(NoopSandbox):
    """macOS Seatbelt sandbox.

    Falls back to :class:`NoopSandbox` if ``sandbox-exec`` is unavailable.
    """

    def __init__(self, config: SandboxConfig) -> None:
        super().__init__(config)
        self._sbpl: str = _generate_sbpl(
            workspace=config.workspace_root,
            tmpdir=Path(os.environ.get("TMPDIR", "/tmp")).resolve(),
            writable=True,
            network=config.network,
        )
        if not shutil.which(_SEATBELT_BIN):
            logger.warning(
                "sandbox-exec not found at {}; SeatbeltSandbox will run commands without OS-level isolation",
                _SEATBELT_BIN,
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

        cmd = [
            _SEATBELT_BIN, "-p", self._sbpl, "--",
            sys.executable, "-m", "agent.sandbox.worker",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
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
            # Fallback to NoopSandbox behaviour
            return await super().exec_command(command, cwd=cwd, timeout=timeout)

        import asyncio

        cmd = [_SEATBELT_BIN, "-p", self._sbpl, "--", "sh", "-c", command]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
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
