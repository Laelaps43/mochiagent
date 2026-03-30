"""Sandboxed worker subprocess entry point.

Usage::

    python -m agent.sandbox.worker

The worker reads a single :class:`WorkerRequest` JSON line from *stdin*,
dynamically loads and executes the requested tool, then writes a
:class:`WorkerResponse` JSON line to *stdout* and exits.

Inside OS-level sandboxes (seatbelt / bwrap) the worker process is
restricted by the OS — file and network access outside the workspace is
denied at the kernel level.  The worker also creates an internal
:class:`NoopSandbox` so that tools calling ``get_sandbox()`` receive a
non-``None`` value.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import traceback
from pathlib import Path

from agent.sandbox.protocol import WorkerRequest, WorkerResponse


async def _run(request: WorkerRequest) -> WorkerResponse:
    # 1. Dynamic import
    try:
        module = importlib.import_module(request.tool_module)
        tool_cls = getattr(module, request.tool_class)
    except Exception as exc:
        return WorkerResponse(
            success=False,
            error=f"Failed to load {request.tool_module}.{request.tool_class}: {exc}",
        )

    # 2. Instantiate
    try:
        tool = tool_cls(**request.init_args)
    except Exception as exc:
        return WorkerResponse(
            success=False,
            error=f"Failed to instantiate {request.tool_class}: {exc}",
        )

    # 3. Set up internal NoopSandbox so get_sandbox() works
    from agent.sandbox.backends.noop import NoopSandbox
    from agent.sandbox.context import reset_sandbox, set_sandbox
    from agent.sandbox.types import SandboxConfig

    sandbox = NoopSandbox(SandboxConfig(workspace_root=Path(request.workspace_root)))
    set_sandbox(sandbox)

    # 4. Execute
    try:
        result = await tool.execute(**request.execute_args)
        return WorkerResponse(success=True, result=result)
    except Exception as exc:
        return WorkerResponse(
            success=False,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
    finally:
        reset_sandbox()


def _write_response(response: WorkerResponse) -> None:
    _ = sys.stdout.write(response.model_dump_json() + "\n")
    _ = sys.stdout.flush()  # type: ignore[func-returns-value]


def main() -> None:
    raw = sys.stdin.readline()
    if not raw.strip():
        _write_response(WorkerResponse(success=False, error="empty stdin"))
        return

    try:
        request = WorkerRequest.model_validate_json(raw)
    except Exception as exc:
        _write_response(WorkerResponse(success=False, error=f"Invalid request: {exc}"))
        return

    response = asyncio.run(_run(request))
    _write_response(response)


if __name__ == "__main__":
    main()
