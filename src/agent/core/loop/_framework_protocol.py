from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from agent.base_agent import BaseAgent


class FrameworkProtocol(Protocol):
    def get_agent(self, agent_name: str) -> BaseAgent | None: ...
