import asyncio
from pathlib import Path

import pytest

from agent import BaseAgent, Tool, reset_framework, setup, shutdown


class EchoTool(Tool):
    @property
    def name(self) -> str:
        return "echo_ready"

    @property
    def description(self) -> str:
        return "ready check"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self):
        return {"ok": True}


class ReadyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.setup_finished = False

    @property
    def name(self) -> str:
        return "ready_agent"

    @property
    def description(self) -> str:
        return "ready agent"

    @property
    def skill_directory(self) -> Path | None:
        return None

    async def setup(self) -> None:
        await asyncio.sleep(0.01)
        self.register_tool(EchoTool())
        self.setup_finished = True


class FailingAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "failing_agent"

    @property
    def description(self) -> str:
        return "failing agent"

    @property
    def skill_directory(self) -> Path | None:
        return None

    async def setup(self) -> None:
        raise RuntimeError("agent setup failed")


@pytest.mark.asyncio
async def test_setup_waits_for_agent_registration_completion():
    reset_framework()
    agent = ReadyAgent()
    try:
        await setup(agents=[agent])
        assert agent.setup_finished is True
        assert "echo_ready" in set(agent.tool_registry.list_tools())
    finally:
        await shutdown()
        reset_framework()


@pytest.mark.asyncio
async def test_setup_raises_when_agent_setup_fails():
    reset_framework()
    try:
        with pytest.raises(RuntimeError, match="agent setup failed"):
            await setup(agents=[FailingAgent()])
    finally:
        await shutdown()
        reset_framework()
