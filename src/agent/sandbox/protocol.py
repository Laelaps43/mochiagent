"""Worker subprocess communication protocol.

The parent process sends a :class:`WorkerRequest` as a single JSON line to
the worker's *stdin*; the worker replies with a :class:`WorkerResponse` JSON
line on *stdout*.
"""

from __future__ import annotations

from pydantic import BaseModel


class WorkerRequest(BaseModel):
    """Payload sent from the host to the sandboxed worker subprocess."""

    tool_module: str
    """Fully-qualified module path, e.g. ``"agent.common.tools.read_file_tool"``."""

    tool_class: str
    """Class name inside *tool_module*, e.g. ``"ReadFileTool"``."""

    init_args: dict[str, object]
    """Constructor kwargs — empty dict for stateless tools."""

    execute_args: dict[str, object]
    """Arguments forwarded to ``tool.execute(**execute_args)``."""

    workspace_root: str
    """Absolute path the worker's internal NoopSandbox should use as root."""


class WorkerResponse(BaseModel):
    """Payload returned from the worker to the host."""

    success: bool
    result: object | None = None
    error: str | None = None
