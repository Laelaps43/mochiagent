from __future__ import annotations

import pytest

from agent.core.compression import DefaultContextCompactor
from agent.core.compression.types import CompactorRunOptions
from agent.core.message import UserTextPart
from agent.core.session.context import SessionContext
from agent.types import ContextBudget, LLMConfig


class _ProviderStub:
    def __init__(self, summary: str = "summary") -> None:
        self.summary = summary
        self.calls = 0

    async def complete(self, messages, tools=None, **kwargs):
        self.calls += 1
        return {"content": self.summary}


@pytest.mark.asyncio
async def test_default_compactor_pre_call_applies_and_keeps_latest_user_last():
    context = SessionContext(session_id="s1", model_profile_id="openai:mock", agent_name="a1")
    context.build_user_message(parts=[UserTextPart(text="old user " + ("x" * 500))])
    context.build_user_message(parts=[UserTextPart(text="latest user question")])

    compactor = DefaultContextCompactor()
    cfg = LLMConfig(
        adapter="openai_compatible",
        provider="openai",
        model="mock",
        context_window_tokens=100,
    )
    result = await compactor.run(
        session_context=context,
        budget=ContextBudget(used_tokens=0),
        llm_config=cfg,
        llm_provider=_ProviderStub("compact summary"),
        options=CompactorRunOptions(stage="pre_call"),
    )

    assert result.applied is True
    assert context.messages[-1].role == "user"
    assert context.messages[-1].parts[0].text == "latest user question"
    assert any(
        msg.parts[0].text.startswith("COMPACTION_SUMMARY\ncompact summary") for msg in context.messages
    )


@pytest.mark.asyncio
async def test_default_compactor_overflow_stage_forces_compaction():
    context = SessionContext(session_id="s2", model_profile_id="openai:mock", agent_name="a1")
    context.build_user_message(parts=[UserTextPart(text="latest")])

    provider = _ProviderStub("overflow summary")
    compactor = DefaultContextCompactor()
    cfg = LLMConfig(adapter="openai_compatible", provider="openai", model="mock")
    result = await compactor.run(
        session_context=context,
        budget=ContextBudget(used_tokens=1),
        llm_config=cfg,
        llm_provider=provider,
        options=CompactorRunOptions(stage="overflow_error"),
    )

    assert result.applied is True
    assert result.reason == "overflow_error"
    assert provider.calls == 1
