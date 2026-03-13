"""
Tool policy engine.

Current stage:
- Global allow/deny by tool name.
- deny has higher priority than allow.
- Empty allow list means allow all (except denied).

Designed for future extension:
- agent/session-scoped rules
- pattern-based rules
- approval hooks
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from agent.config.tools import ToolPolicyConfig


class PolicyDecision(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    allowed: bool
    reason: str


class ToolPolicyEngine:
    def __init__(self, config: ToolPolicyConfig):
        self.config: ToolPolicyConfig = config.normalized()

    def evaluate(self, tool_name: str) -> PolicyDecision:
        normalized = (tool_name or "").strip().lower()

        if normalized in (self.config.deny or set()):
            return PolicyDecision(
                allowed=False,
                reason=f"tool '{tool_name}' is denied by TOOLS_POLICY_DENY",
            )

        # allow 为空：默认允许
        if not self.config.allow:
            return PolicyDecision(allowed=True, reason="allowed by default")

        if normalized in self.config.allow:
            return PolicyDecision(allowed=True, reason="tool in TOOLS_POLICY_ALLOW")

        return PolicyDecision(
            allowed=False,
            reason=f"tool '{tool_name}' is not in TOOLS_POLICY_ALLOW",
        )
