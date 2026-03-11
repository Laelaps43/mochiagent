from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.core.bus import MessageBus
from agent.core.compression import CompactionPayload
from agent.core.loop import AgentEventLoop
from agent.core.message import UserTextInput as UserTextPart
from agent.core.loop.turn_result import LLMTurnResult
from agent.core.runtime import StrategyKind
from agent.core.session.context import SessionContext
from agent.types import ContextBudget, EventType, LLMConfig, LLMStreamChunk, ProviderUsage, SessionState, TokenUsage, ToolCallPayload, ToolFunctionPayload


class _SessionManagerStub:
    def __init__(self, context: SessionContext):
        self._context = context
        self.finish_calls: list[dict] = []
        self.states: list[SessionState] = []
        self.metadata_save_calls = 0
        self.emitted_events = []

    async def get_session(self, session_id: str) -> SessionContext:
        return self._context

    async def finish_assistant_message(
        self,
        session_id: str,
        tokens: TokenUsage | None = None,
        finish: str = "stop",
    ) -> None:
        self.finish_calls.append(
            {
                "session_id": session_id,
                "tokens": tokens or TokenUsage(),
                "finish": finish,
            }
        )
        self._context.finish_current_message(tokens=tokens, finish=finish)

    async def emit_to_session_listeners(self, session_id: str, event) -> None:
        self.emitted_events.append(event)
        return None

    async def update_state(self, session_id: str, new_state: SessionState) -> None:
        self.states.append(new_state)
        self._context.update_state(new_state)

    async def save_session_metadata(self, session_id: str) -> None:
        self.metadata_save_calls += 1


class _LLMNoToolStub:
    async def stream_chat(self, messages, tools=None):
        yield LLMStreamChunk(content="hello")
        yield LLMStreamChunk(finish_reason="stop")


class _LLMNoToolWithUsageStub:
    async def stream_chat(self, messages, tools=None):
        yield LLMStreamChunk(content="hello")
        yield LLMStreamChunk(
            usage=ProviderUsage(
                input_tokens=120,
                output_tokens=30,
                reasoning_tokens=7,
            ),
            finish_reason="stop",
        )


class _LLMThinkingStub:
    async def stream_chat(self, messages, tools=None):
        yield LLMStreamChunk(thinking="step-1")
        yield LLMStreamChunk(thinking=" step-2")
        yield LLMStreamChunk(content="hello")
        yield LLMStreamChunk(finish_reason="stop")


class _AdapterRegistryStub:
    def get(self, config):
        return _LLMNoToolStub()


async def _noop_compaction(kind, **_kwargs):
    assert kind == StrategyKind.CONTEXT_COMPACTION
    return CompactionPayload.invalid(stage="pre_call", reason="skip", name="noop")


@pytest.mark.asyncio
async def test_call_llm_turn_uses_finish_assistant_message_for_persistence():
    context = SessionContext(
        session_id="sess_no_tool",
        model_profile_id="safe",
        agent_name="agent_stub",
    )
    context.build_user_message(parts=[UserTextPart(text="hi")])
    session_manager = _SessionManagerStub(context)

    _agent_context = SimpleNamespace(
        resolve_llm_config_for_agent=lambda _agent, _profile: LLMConfig(
            adapter="openai_compatible",
            provider="openai",
            model="mock",
        ),
        strategy_manager=SimpleNamespace(run=_noop_compaction),
    )
    framework = SimpleNamespace(
        get_agent=lambda _: SimpleNamespace(
            tool_registry=SimpleNamespace(get_definitions=lambda: []),
            get_system_prompt=lambda _ctx: None,
            context=_agent_context,
        ),
    )

    loop = AgentEventLoop(
        bus=MessageBus(),
        session_manager=session_manager,
        adapter_registry=_AdapterRegistryStub(),
        framework=framework,
    )

    await loop._conversation_loop("sess_no_tool")

    assert len(session_manager.finish_calls) == 1
    assert session_manager.finish_calls[0]["session_id"] == "sess_no_tool"
    assert session_manager.finish_calls[0]["finish"] == "stop"


@pytest.mark.asyncio
async def test_call_llm_turn_sets_context_budget_without_usage():
    context = SessionContext(
        session_id="sess_budget_zero",
        model_profile_id="safe",
        agent_name="agent_stub",
    )
    context.build_user_message(parts=[UserTextPart(text="hi")])
    session_manager = _SessionManagerStub(context)

    _agent_context = SimpleNamespace(
        resolve_llm_config_for_agent=lambda _agent, _profile: LLMConfig(
            adapter="openai_compatible",
            provider="openai",
            model="mock",
            context_window_tokens=1000,
        ),
        strategy_manager=SimpleNamespace(run=_noop_compaction),
    )
    framework = SimpleNamespace(
        get_agent=lambda _: SimpleNamespace(
            tool_registry=SimpleNamespace(get_definitions=lambda: []),
            get_system_prompt=lambda _ctx: None,
            context=_agent_context,
        ),
    )

    loop = AgentEventLoop(
        bus=MessageBus(),
        session_manager=session_manager,
        adapter_registry=_AdapterRegistryStub(),
        framework=framework,
    )

    result = await loop._llm_turn_handler.run("sess_budget_zero")

    assert result.tokens == TokenUsage(input=0, output=0, reasoning=0)
    budget = result.context_budget
    assert budget.source == "estimated"
    assert budget.total_tokens == 1000
    assert budget.used_tokens == 0
    assert budget.remaining_tokens == 1000
    assert session_manager.metadata_save_calls == 1


@pytest.mark.asyncio
async def test_call_llm_turn_sets_context_budget_from_provider_usage():
    context = SessionContext(
        session_id="sess_budget_provider",
        model_profile_id="safe",
        agent_name="agent_stub",
    )
    context.build_user_message(parts=[UserTextPart(text="hi")])
    session_manager = _SessionManagerStub(context)

    _agent_context = SimpleNamespace(
        resolve_llm_config_for_agent=lambda _agent, _profile: LLMConfig(
            adapter="openai_compatible",
            provider="openai",
            model="mock",
            context_window_tokens=2000,
        ),
        strategy_manager=SimpleNamespace(run=_noop_compaction),
    )
    framework = SimpleNamespace(
        get_agent=lambda _: SimpleNamespace(
            tool_registry=SimpleNamespace(get_definitions=lambda: []),
            get_system_prompt=lambda _ctx: None,
            context=_agent_context,
        ),
    )

    loop = AgentEventLoop(
        bus=MessageBus(),
        session_manager=session_manager,
        adapter_registry=SimpleNamespace(get=lambda _cfg: _LLMNoToolWithUsageStub()),
        framework=framework,
    )

    result = await loop._llm_turn_handler.run("sess_budget_provider")

    assert result.tokens == TokenUsage(input=120, output=30, reasoning=7)
    budget = result.context_budget
    assert budget.source == "provider"
    assert budget.used_tokens == 157
    assert budget.remaining_tokens == 1843
    assert session_manager.metadata_save_calls == 1


@pytest.mark.asyncio
async def test_call_llm_turn_exposes_thinking_content():
    context = SessionContext(
        session_id="sess_thinking",
        model_profile_id="safe",
        agent_name="agent_stub",
    )
    context.build_user_message(parts=[UserTextPart(text="hi")])
    session_manager = _SessionManagerStub(context)

    _agent_context = SimpleNamespace(
        resolve_llm_config_for_agent=lambda _agent, _profile: LLMConfig(
            adapter="openai_compatible",
            provider="openai",
            model="mock",
        ),
        strategy_manager=SimpleNamespace(run=_noop_compaction),
    )
    framework = SimpleNamespace(
        get_agent=lambda _: SimpleNamespace(
            tool_registry=SimpleNamespace(get_definitions=lambda: []),
            get_system_prompt=lambda _ctx: None,
            context=_agent_context,
        ),
    )

    loop = AgentEventLoop(
        bus=MessageBus(),
        session_manager=session_manager,
        adapter_registry=SimpleNamespace(get=lambda _cfg: _LLMThinkingStub()),
        framework=framework,
    )

    result = await loop._llm_turn_handler.run("sess_thinking")

    assert result.thinking == "step-1 step-2"
    assert result.content == "hello"
    thinking_events = [
        event
        for event in session_manager.emitted_events
        if event.type == EventType.LLM_THINKING
    ]
    assert len(thinking_events) == 2
    assert thinking_events[0].data["thinking"] == "step-1"
    assert thinking_events[1].data["thinking"] == " step-2"


@pytest.mark.asyncio
async def test_conversation_loop_tool_branch_uses_finish_assistant_message():
    context = SessionContext(
        session_id="sess_tool",
        model_profile_id="safe",
        agent_name="agent_stub",
    )
    context.build_user_message(parts=[UserTextPart(text="hi")])
    session_manager = _SessionManagerStub(context)

    framework = SimpleNamespace(
        resolve_llm_config_for_agent=lambda _agent, _profile: LLMConfig(
            adapter="openai_compatible",
            provider="openai",
            model="mock",
        ),
        get_agent=lambda _: None,
        strategy_manager=SimpleNamespace(run=_noop_compaction),
    )
    loop = AgentEventLoop(
        bus=MessageBus(),
        session_manager=session_manager,
        adapter_registry=_AdapterRegistryStub(),
        framework=framework,
    )

    call_count = {"n": 0}

    async def _fake_call_llm_turn(session_id: str):
        if call_count["n"] == 0:
            call_count["n"] += 1
            context.build_assistant_message(
                parent_id=context.messages[-1].message_id,
                provider_id="openai",
                model_id="mock",
            )
            return LLMTurnResult(
                content="",
                thinking="",
                tool_calls=[
                    ToolCallPayload(
                        id="call_1",
                        function=ToolFunctionPayload(name="echo", arguments="{}"),
                    )
                ],
                finish_reason="tool_calls",
                tokens=TokenUsage(),
                context_budget=ContextBudget(),
                context_compaction=CompactionPayload.invalid(stage="pre_call"),
                context_compaction_events=[],
                message_id=context.current_message.message_id if context.current_message else "",
            )

        await session_manager.finish_assistant_message(
            session_id=session_id, tokens=TokenUsage(), finish="stop",
        )
        return LLMTurnResult(
            content="done",
            thinking="",
            tool_calls=[],
            finish_reason="stop",
            tokens=TokenUsage(),
            context_budget=ContextBudget(),
            context_compaction=CompactionPayload.invalid(stage="pre_call"),
            context_compaction_events=[],
            message_id="msg_done",
        )

    async def _fake_execute_tools(session_id: str, tool_calls: list):
        return []

    loop._llm_turn_handler.run = _fake_call_llm_turn  # type: ignore[method-assign]
    loop._execute_tools = _fake_execute_tools  # type: ignore[method-assign]

    await loop._conversation_loop("sess_tool")

    assert len(session_manager.finish_calls) >= 1
