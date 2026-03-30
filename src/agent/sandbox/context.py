"""Sandbox context variable — shared between sandbox and tools.

This module lives in the sandbox package to avoid circular imports between
``agent.sandbox`` and ``agent.common.tools``.

Uses ``object`` as the ContextVar type to avoid importing ``Sandbox`` and
creating a cycle (abc → context → abc).  The actual type is always
``Sandbox | None`` at runtime.
"""

from __future__ import annotations

import contextvars

__all__ = ["get_sandbox", "set_sandbox", "reset_sandbox"]

_sandbox_var: contextvars.ContextVar[object] = contextvars.ContextVar(
    "_sandbox_var", default=None
)


def set_sandbox(sandbox: object) -> None:
    _ = _sandbox_var.set(sandbox)


def get_sandbox() -> object:
    return _sandbox_var.get()


def reset_sandbox() -> None:
    _ = _sandbox_var.set(None)
