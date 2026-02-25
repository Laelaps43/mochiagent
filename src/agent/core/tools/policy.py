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

from dataclasses import dataclass


def _parse_name_list(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {item.strip().lower() for item in raw.split(",") if item and item.strip()}


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class ToolPolicyConfig:
    allow: set[str] | None = None
    deny: set[str] | None = None

    def normalized(self) -> "ToolPolicyConfig":
        return ToolPolicyConfig(
            allow={x.lower() for x in (self.allow or set())},
            deny={x.lower() for x in (self.deny or set())},
        )

    @classmethod
    def from_csv(
        cls,
        *,
        allow_csv: str | None = None,
        deny_csv: str | None = None,
    ) -> "ToolPolicyConfig":
        return cls(
            allow=_parse_name_list(allow_csv),
            deny=_parse_name_list(deny_csv),
        )


class ToolPolicyEngine:
    def __init__(self, config: ToolPolicyConfig):
        self.config = config.normalized()

    def evaluate(self, tool_name: str) -> PolicyDecision:
        normalized = (tool_name or "").strip().lower()

        if normalized in self.config.deny:
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
