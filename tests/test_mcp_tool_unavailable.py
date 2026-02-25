import asyncio
from types import SimpleNamespace

import pytest

from agent.common.tools.mcp import MCPToolWrapper
from agent.core.mcp.manager import MCPManager
from agent.core.tools import ToolRegistry


class _SlowSession:
    async def call_tool(self, name, arguments=None):
        await asyncio.sleep(0.05)
        return SimpleNamespace(content=[])


@pytest.mark.asyncio
async def test_mcp_wrapper_enters_cooldown_after_threshold():
    registry = ToolRegistry()
    manager = MCPManager(registry=registry, default_timeout=30)
    manager.register_server(
        "docs",
        {
            "failureThreshold": 2,
            "cooldownSec": 60,
            "toolTimeout": 0,
        },
    )
    tool_def = SimpleNamespace(
        name="search_docs",
        description="",
        inputSchema={"type": "object", "properties": {}},
    )
    wrapper = MCPToolWrapper(
        session=_SlowSession(),
        server_name="docs",
        tool_def=tool_def,
        timeout=0,
        manager=manager,
    )

    result_1 = await wrapper.execute()
    assert "timeout" in result_1.lower()
    assert manager.snapshot()["docs"]["status"] == "degraded"

    result_2 = await wrapper.execute()
    assert "timeout" in result_2.lower()
    assert manager.snapshot()["docs"]["status"] == "failed"

    # Failed status should short-circuit until cooldown window passes.
    result_3 = await wrapper.execute()
    assert "cooling down" in result_3.lower()
