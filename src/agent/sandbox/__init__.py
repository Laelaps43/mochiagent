"""Sandbox subsystem — pluggable security isolation for tool execution.

Public API::

    from agent.sandbox import (
        Sandbox,
        SandboxConfig,
        SandboxDecision,
        NoopSandbox,
        SeatbeltSandbox,
        BwrapSandbox,
        create_sandbox,
    )
"""

from agent.sandbox.abc import Sandbox
from agent.sandbox.backends.noop import NoopSandbox
from agent.sandbox.context import get_sandbox, reset_sandbox, set_sandbox
from agent.sandbox.factory import create_sandbox
from agent.sandbox.protocol import WorkerRequest, WorkerResponse
from agent.sandbox.types import SandboxConfig, SandboxDecision

__all__ = [
    "Sandbox",
    "SandboxConfig",
    "SandboxDecision",
    "NoopSandbox",
    "WorkerRequest",
    "WorkerResponse",
    "create_sandbox",
    "get_sandbox",
    "set_sandbox",
    "reset_sandbox",
]

# Lazy imports — only load when accessed to avoid import errors on
# platforms where a backend isn't available.


def __getattr__(name: str) -> object:
    if name == "SeatbeltSandbox":
        from agent.sandbox.backends.seatbelt import SeatbeltSandbox
        return SeatbeltSandbox
    if name == "BwrapSandbox":
        from agent.sandbox.backends.bwrap import BwrapSandbox
        return BwrapSandbox
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
