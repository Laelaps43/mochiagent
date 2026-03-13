from __future__ import annotations

from typing import Protocol

from agent.base_agent import BaseAgent


class FrameworkProtocol(Protocol):
    def get_agent(self, agent_name: str) -> BaseAgent | None: ...
