"""Schema-annotation inspector — extracted from ToolSecurityGuard.

Reads ``x-workspace-path``, ``x-workspace-cwd`` and ``x-shell-command``
annotations from a tool's ``parameters_schema`` and delegates to the
sandbox's ``check_path`` / command-check methods.

This is a *private* module used by all sandbox backends.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from agent.sandbox.types import SandboxDecision

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


async def inspect_tool_call(
    tool: object,
    arguments: Mapping[str, object],
    *,
    check_path: Callable[[str], Awaitable[SandboxDecision]],
    check_command: Callable[[str, Mapping[str, object]], Awaitable[SandboxDecision]],
) -> SandboxDecision:
    """Inspect a tool's schema annotations and run security checks.

    This is the shared implementation that was previously in
    ``ToolSecurityGuard.validate_tool_call()``.
    """
    raw_schema = getattr(tool, "parameters_schema", None)
    schema = cast(dict[str, object], raw_schema) if isinstance(raw_schema, dict) else {}
    raw_props = schema.get("properties")
    properties = cast(dict[str, object], raw_props) if isinstance(raw_props, dict) else {}

    for key, raw_prop in properties.items():
        if not isinstance(raw_prop, dict):
            continue
        prop = cast(dict[str, object], raw_prop)
        value = arguments.get(key)
        if value is None:
            continue

        # Path annotation
        if (prop.get("x-workspace-path") or prop.get("x-workspace-cwd")) and isinstance(value, str):
            decision = await check_path(value)
            if not decision.allowed:
                return decision

        # Shell-command annotation
        if prop.get("x-shell-command") and isinstance(value, str):
            decision = await check_command(value, arguments)
            if not decision.allowed:
                return decision

    return SandboxDecision(allowed=True, reason="allowed")
