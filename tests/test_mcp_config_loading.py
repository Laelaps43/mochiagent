from pathlib import Path

import pytest

from agent.base_agent import BaseAgent


class DummyAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "dummy"

    @property
    def skill_directory(self) -> Path | None:
        return None

    async def setup(self) -> None:
        return


@pytest.mark.asyncio
async def test_register_mcp_tools_ignores_missing_config(tmp_path: Path):
    agent = DummyAgent()
    await agent.register_mcp_tools(tmp_path / "missing.json")
    assert agent.tool_registry.list_tools() == []


@pytest.mark.asyncio
async def test_register_mcp_tools_ignores_empty_servers(tmp_path: Path):
    cfg = tmp_path / "mcp.json"
    cfg.write_text('{"mcpServers": {}}', encoding="utf-8")
    agent = DummyAgent()
    await agent.register_mcp_tools(cfg)
    assert agent.tool_registry.list_tools() == []
