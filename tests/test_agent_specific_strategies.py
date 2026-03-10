from __future__ import annotations

import pytest

from agent.core.compression import CompactionStage
from agent.core.runtime import StrategyKind
from agent.framework import AgentFramework
from agent.types import ContextBudget, LLMConfig, ToolResult


class _CompactorA:
    async def run(self, *, session_context, budget, llm_config, llm_provider, options):
        return _Result(applied=False, reason="A")


class _CompactorB:
    async def run(self, *, session_context, budget, llm_config, llm_provider, options):
        return _Result(applied=True, reason="B")


class _PostprocessorA:
    async def process(self, *, session_id, tool_result, tool_arguments, storage):
        tool_result.summary = "A"
        return tool_result


class _PostprocessorB:
    async def process(self, *, session_id, tool_result, tool_arguments, storage):
        tool_result.summary = "B"
        return tool_result


class _Result:
    def __init__(self, *, applied: bool, reason: str):
        self.applied = applied
        self.reason = reason
        self.metadata: dict = {}
        self.stats: dict = {}
        self.artifacts: list = []

    def to_dict(self):
        return {
            "applied": self.applied,
            "reason": self.reason,
            "metadata": self.metadata,
            "stats": self.stats,
            "artifacts": self.artifacts,
        }


@pytest.mark.asyncio
async def test_agent_specific_context_compactor_override():
    framework = AgentFramework()
    framework.strategy_manager.register(StrategyKind.CONTEXT_COMPACTION, "a", lambda _opts: _CompactorA())
    framework.strategy_manager.register(StrategyKind.CONTEXT_COMPACTION, "b", lambda _opts: _CompactorB())
    framework.strategy_manager.set_agent(StrategyKind.CONTEXT_COMPACTION, "agent_a", "a")
    framework.strategy_manager.set_agent(StrategyKind.CONTEXT_COMPACTION, "agent_b", "b")

    cfg = LLMConfig(adapter="openai_compatible", provider="openai", model="mock")
    global_result = await framework.strategy_manager.run(
        StrategyKind.CONTEXT_COMPACTION,
        session_context=object(),
        budget=ContextBudget(),
        llm_config=cfg,
        llm_provider=object(),
        stage=CompactionStage.PRE_CALL,
        agent_name="agent_a",
    )
    agent_result = await framework.strategy_manager.run(
        StrategyKind.CONTEXT_COMPACTION,
        session_context=object(),
        budget=ContextBudget(),
        llm_config=cfg,
        llm_provider=object(),
        stage=CompactionStage.PRE_CALL,
        agent_name="agent_b",
    )

    assert global_result.name == "a"
    assert global_result.reason == "A"
    assert agent_result.name == "b"
    assert agent_result.reason == "B"


@pytest.mark.asyncio
async def test_agent_specific_tool_postprocessor_override():
    framework = AgentFramework()
    framework.strategy_manager.register(
        StrategyKind.TOOL_RESULT_POSTPROCESS, "a", lambda _opts: _PostprocessorA()
    )
    framework.strategy_manager.register(
        StrategyKind.TOOL_RESULT_POSTPROCESS, "b", lambda _opts: _PostprocessorB()
    )
    framework.strategy_manager.set_agent(StrategyKind.TOOL_RESULT_POSTPROCESS, "agent_a", "a")
    framework.strategy_manager.set_agent(StrategyKind.TOOL_RESULT_POSTPROCESS, "agent_b", "b")

    base = ToolResult(tool_call_id="c1", tool_name="exec", result="x", success=True)
    out_a = await framework.strategy_manager.run(
        StrategyKind.TOOL_RESULT_POSTPROCESS,
        agent_name="agent_a",
        session_id="s1",
        tool_result=base.model_copy(deep=True),
        tool_arguments={},
        storage=object(),
    )
    out_b = await framework.strategy_manager.run(
        StrategyKind.TOOL_RESULT_POSTPROCESS,
        agent_name="agent_b",
        session_id="s2",
        tool_result=base.model_copy(deep=True),
        tool_arguments={},
        storage=object(),
    )

    assert out_a.summary == "A"
    assert out_b.summary == "B"
