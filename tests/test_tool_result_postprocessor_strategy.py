from __future__ import annotations

import pytest

from agent.core.runtime import StrategyKind
from agent.core.storage import MemoryStorage
from agent.framework import AgentFramework
from agent.types import ToolResult


class _CustomPostprocessor:
    async def process(
        self,
        *,
        session_id: str,
        tool_result: ToolResult,
        tool_arguments: dict,
        storage,
    ) -> ToolResult:
        tool_result.summary = f"custom:{tool_result.tool_name}"
        return tool_result


@pytest.mark.asyncio
async def test_framework_custom_tool_result_postprocessor_is_used():
    framework = AgentFramework()
    framework.strategy_manager.register(
        StrategyKind.TOOL_RESULT_POSTPROCESS,
        "custom", lambda _opts: _CustomPostprocessor()
    )
    framework.strategy_manager.set_agent(
        StrategyKind.TOOL_RESULT_POSTPROCESS,
        "agent_custom",
        "custom",
    )

    result = ToolResult(
        tool_call_id="call_1",
        tool_name="read_file",
        result={"content": "hello"},
        success=True,
    )

    out = await framework.strategy_manager.run(
        StrategyKind.TOOL_RESULT_POSTPROCESS,
        agent_name="agent_custom",
        session_id="s1",
        tool_result=result,
        tool_arguments={"path": "README.md"},
        storage=MemoryStorage(),
    )
    assert out.summary == "custom:read_file"
